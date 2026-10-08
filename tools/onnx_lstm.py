"""The exported networks with their recurrence as ONNX's LSTM op, for the
graphics card.

tools/export_onnx.py writes each LSTM layer as a Loop: one iteration a frame,
~47 small ops each. On a processor that is fine; on a card every op is a
kernel launch and the Loop's control runs on the host, so a game's value
read took longer on an RTX 4070 than on four processor threads (2.9 s
against 5.1 s a game; measured 2026-10-02). ONNX's LSTM op is one kernel
call for the whole sequence (cuDNN on CUDA): the same game reads in 0.14 s.

This writes <name>.lstm.onnx next to <name>.onnx (the .onnx.json stays the
one both read): every Loop that is an LSTM cell - x @ Wx precomputed outside
it by an Einsum, gates = that + h @ Rh + bias split i, f, g, o, the state
put back to its start where `reset` says - becomes an LSTM node with the
same weights in ONNX's gate order (i, o, f, c), fed the Einsum's input.

What the LSTM op cannot do is reset a row in the middle of a sequence, so
the variant drops the `reset` input: onnx_models.Model zeroes a row's state
itself where reset is set on the first frame (every reset value here is
zero - this checks), and refuses a reset anywhere later. A game's read only
ever resets at its first frame.

Checked before writing: on seeded inputs (onnx_models.check_feed) the
variant gives the original's outputs on the processor to 1e-4.

  ..\\export-env\\Scripts\\python tools\\onnx_lstm.py PATH\\vf_grouped_tx1x1024.onnx [...]

Needs the `onnx` package (export-env has it; the app's Python does not).
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
  sys.path.insert(0, HERE)

TOL = 1e-4


def variant_path(path):
  return path[:-len('.onnx')] + '.lstm.onnx'


def _values(graph, body=None):
  """{name: array} of a graph's initializers and Constant nodes (and the
  outer graph's, for a Loop body that reads them)."""
  from onnx import numpy_helper
  out = {i.name: numpy_helper.to_array(i) for i in graph.initializer}
  for n in graph.node:
    if n.op_type == 'Constant':
      out[n.output[0]] = numpy_helper.to_array(n.attribute[0].t)
  if body is not None:
    out.update(_values(body))
  return out


def _cell(loop, outer):
  """The LSTM inside a Loop: (Rh [H, 4H], bias [4H], reset c, reset h), or
  ValueError when the body is something else."""
  body = loop.attribute[0].g
  vals = _values(outer, body)
  prod = {o: n for n in body.node for o in n.output}
  ops = {n.op_type for n in body.node}
  if not {'Gemm', 'Split', 'Sigmoid', 'Tanh'} <= ops:
    raise ValueError('%s is not an LSTM cell' % loop.name)
  gemm = next(n for n in body.node if n.op_type == 'Gemm')
  if any(a.name in ('transA', 'transB') and a.i for a in gemm.attribute):
    raise ValueError('%s: a transposed Gemm' % loop.name)
  beta = next((a.f for a in gemm.attribute if a.name == 'beta'), 1.0)
  rh = vals[gemm.input[1]]
  extra = vals[gemm.input[2]] * beta if len(gemm.input) > 2 and beta else 0.0
  add = next(n for n in body.node if n.op_type == 'Add' and gemm.output[0] in n.input)
  other = [i for i in add.input if i != gemm.output[0]][0]
  bias = (vals[other] + extra).reshape(-1).astype(np.float32)
  # the state after a reset: Where(reset, Expand(const), state)
  resets = []
  for n in body.node:
    if n.op_type == 'Where':
      exp = prod.get(n.input[1])
      src = exp.input[0] if exp is not None and exp.op_type == 'Expand' else n.input[1]
      resets.append((n.input[2], vals[src]))
  # body inputs: iter, cond, c (state_2k), h (state_2k+1), x seq, reset seq
  c_in, h_in = body.input[2].name, body.input[3].name
  rv = dict(resets)
  if c_in not in rv or h_in not in rv:
    raise ValueError('%s: no reset of its state' % loop.name)
  if gemm.input[0] not in prod or prod[gemm.input[0]].op_type != 'Where':
    raise ValueError('%s: the recurrence does not read the reset state' % loop.name)
  return rh, bias, rv[c_in], rv[h_in]


def _gates(a, H):
  """i, f, g, o (the export's Split order) as ONNX's i, o, f, c."""
  b = [a[..., k * H:(k + 1) * H] for k in range(4)]
  return [b[0], b[3], b[1], b[2]]


def to_lstm(src, dst=None):
  """Write src's LSTM variant; returns its path."""
  import onnx
  from onnx import helper, numpy_helper
  dst = dst or variant_path(src)
  m = onnx.load(src)
  g = m.graph
  outer = _values(g)
  loops = [n for n in g.node if n.op_type == 'Loop']
  if not loops:
    raise ValueError('%s has no Loop to replace' % os.path.basename(src))
  order = [n.name for n in g.node]
  skip, new_at, extra = set(), {}, []
  for k, loop in enumerate(loops):
    rh, bias, c0, h0 = _cell(loop, g)
    if np.abs(c0).max() or np.abs(h0).max():
      raise ValueError('%s: a reset to a state that is not zero' % loop.name)
    H = rh.shape[0]
    x_seq = loop.input[4]
    ein = next(n for n in g.node if x_seq in n.output)
    if ein.op_type != 'Einsum':
      raise ValueError('%s: its inputs are not an Einsum projection' % loop.name)
    wx = outer.get(ein.input[1])
    cat = None
    if wx is None:
      cat = next(n for n in g.node if ein.input[1] in n.output)
      wx = np.concatenate([outer[x] for x in cat.input], axis=1)
      skip.add(cat.name)
    W = np.concatenate([b.T for b in _gates(wx, H)], 0)[None].astype(np.float32)
    R = np.concatenate([b.T for b in _gates(rh, H)], 0)[None].astype(np.float32)
    B = np.concatenate([np.concatenate(_gates(bias, H)), np.zeros(4 * H, np.float32)])[None]
    c_state, h_state = loop.input[2], loop.input[3]
    p = 'lstm%d_' % k
    extra += [numpy_helper.from_array(W, p + 'W'), numpy_helper.from_array(R, p + 'R'),
              numpy_helper.from_array(B.astype(np.float32), p + 'B')]
    new_at[loop.name] = [
        helper.make_node('Unsqueeze', [h_state, 'lstm_ax0'], [p + 'h0']),
        helper.make_node('Unsqueeze', [c_state, 'lstm_ax0'], [p + 'c0']),
        helper.make_node('LSTM', [ein.input[0], p + 'W', p + 'R', p + 'B', '', p + 'h0', p + 'c0'],
                         [p + 'Y', p + 'Yh', p + 'Yc'], hidden_size=H),
        helper.make_node('Squeeze', [p + 'Y', 'lstm_ax1'], [loop.output[4]]),
        helper.make_node('Squeeze', [p + 'Yc', 'lstm_ax0'], [loop.output[0]]),
        helper.make_node('Squeeze', [p + 'Yh', 'lstm_ax0'], [loop.output[1]]),
    ]
    skip |= {loop.name, ein.name}
  nodes = []
  for n in g.node:
    if n.name in new_at:
      nodes += new_at[n.name]
    if n.name not in skip:
      nodes.append(n)
  outs = {o.name for o in g.output}
  while True:                         # what only fed the Loops (trip counts) goes
    used = {i for n in nodes for i in n.input} | outs
    kept = [n for n in nodes if any(o in used for o in n.output)]
    if len(kept) == len(nodes):
      break
    nodes = kept
  extra += [numpy_helper.from_array(np.array([0], np.int64), 'lstm_ax0'),
            numpy_helper.from_array(np.array([1], np.int64), 'lstm_ax1')]
  inits = [i for i in g.initializer if i.name in used] + extra
  inputs = [i for i in g.input if i.name in used]
  g.ClearField('node')
  g.node.extend(nodes)
  g.ClearField('initializer')
  g.initializer.extend(inits)
  g.ClearField('input')
  g.input.extend(inputs)
  # which export this was made from (onnx_models._variant_matches): one left
  # behind by an older export is not used
  import json
  with open(src + '.json', encoding='utf-8') as f:
    exported = json.load(f).get('exported')
  for k, v in (('from_exported', str(exported)), ('from_bytes', str(os.path.getsize(src)))):
    p = m.metadata_props.add()
    p.key, p.value = k, v
  onnx.checker.check_model(m)
  tmp = dst + '.tmp'
  onnx.save(m, tmp)
  worst = compare(src, tmp)
  if worst > TOL:
    os.remove(tmp)
    raise ValueError('the variant is %.3g off the original' % worst)
  os.replace(tmp, dst)
  print('%s: %d Loop%s as LSTM, %.2g off the original on the processor'
        % (os.path.basename(dst), len(loops), '' if len(loops) == 1 else 's', worst))
  return dst


def compare(src, variant):
  """The largest float output difference between the original and the
  variant on the processor, on seeded inputs whose reset is only at the
  first frame (what the variant supports)."""
  import onnxruntime as ort
  import onnx_models
  so = ort.SessionOptions()
  so.log_severity_level = 3
  a = ort.InferenceSession(src, so, providers=['CPUExecutionProvider'])
  b = ort.InferenceSession(variant, so, providers=['CPUExecutionProvider'])
  names = [o.name for o in b.get_outputs()]
  takes = {i.name for i in b.get_inputs()}
  worst = 0.0
  # a read from its start (the variant is given the zero state the reset
  # would have put), and a read carrying on from a state (no reset at all)
  for start in (True, False):
    feed = onnx_models.check_feed(a)
    if 'reset' not in feed:
      raise ValueError('%s has no reset input' % os.path.basename(src))
    feed['reset'][...] = False
    feed['reset'][0] = start
    want = dict(zip([o.name for o in a.get_outputs()], a.run(None, feed)))
    mine = {k: (np.zeros_like(v) if start and k.startswith('state_') else v)
            for k, v in feed.items() if k in takes}
    for name, x in zip(names, b.run(None, mine)):
      if np.issubdtype(np.asarray(x).dtype, np.floating):
        worst = max(worst, float(np.max(np.abs(np.asarray(x, np.float64) - want[name]))))
  return worst


def main(argv=None):
  argv = sys.argv[1:] if argv is None else argv
  if not argv:
    print(__doc__)
    return 2
  for src in argv:
    to_lstm(src)
  return 0


if __name__ == '__main__':
  sys.exit(main())
