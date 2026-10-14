"""A copy of an export whose weights are stored as float16 (docs/providers.md).

  python tools/onnx_f16.py SRC.onnx DST.onnx [--rounded DST32.onnx] [--min 256]
                           [--variant-of CONVERTED_EXPORT.onnx]

Storage only: each float32 weight of --min values or more is kept as float16
under <name>__f16, with one Cast back to float32 that writes the weight's own
name. onnxruntime folds those Casts when it loads the file, so every kernel
runs in float32 on the rounded weights; the file is about half the size. No
other node, input, output or type is touched (a converter that retypes the
graph is where mistyped Casts come from).

--rounded also writes the same rounded weights stored as float32: the two
files must give the same bits, which is the check that nothing but the
rounding changed.

A weight that float16 cannot hold (past 65,504, or one that would round to
zero or lose more than --rel of itself) stays float32 and is counted. A Loop
body's own weights are moved out to the graph first, so they are stored the
same way and cast once.

--drift holds the float16 file's answer to the original's on the app's fixed
check input (F16_REL, below). tools/refresh_models.py runs both checks on
every file it converts.

Needs onnx and numpy (and onnxruntime for --drift). A developer tool: the app
does not use it.
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
  sys.path.insert(1, HERE)                 # onnx_models, for the fixed check input


def convert(src, dst, rounded=None, least=256, rel=0.01, variant_of=None, exact=False):
  """exact: store a weight as float16 only when float16 holds every value of
  it as it is (a variant made from the rounded twin of a converted export:
  its copies of that export's weights are stored as float16 again, anything
  it worked out of them stays float32, so it computes with exactly the
  converted export's weights)."""
  import onnx
  from onnx import TensorProto, helper, numpy_helper
  m = onnx.load(src)
  g = m.graph
  graph_inputs = {i.name for i in g.input}
  r = onnx.ModelProto()
  r.CopyFrom(m) if rounded else None
  casts, said = [], dict(stored=0, kept_small=0, kept_range=0, kept_inexact=0, bytes16=0,
                         bytes32_left=0, worst_abs=0.0, worst_rel=0.0, body=0)
  rounded_arrays = {}
  taken = {i.name for i in g.initializer} | graph_inputs
  for n in g.node:
    taken.update(n.output)

  def bodies(graph):
    for n in graph.node:
      for at in n.attribute:
        for body in ([at.g] if at.HasField('g') else list(at.graphs)):
          yield body
          yield from bodies(body)

  def store(init, hoisted):
    """The float16 weight and its Cast for one float32 weight, or None."""
    a = numpy_helper.to_array(init)
    if a.size < least:
      said['kept_small'] += 1
      said['bytes32_left'] += a.nbytes
      return None
    h = a.astype(np.float16)
    back = h.astype(np.float32)
    big = np.abs(a) > 1e-3
    off = np.abs(back - a)
    worst_rel = float((off[big] / np.abs(a[big])).max()) if big.any() else 0.0
    if exact and not np.array_equal(back, a, equal_nan=True):
      said['kept_inexact'] += 1
      said['bytes32_left'] += a.nbytes
      return None
    if not np.isfinite(back).all() or worst_rel > rel or (hoisted and init.name in taken):
      said['kept_range'] += 1
      said['bytes32_left'] += a.nbytes
      return None
    said['stored'] += 1
    said['body'] += int(hoisted)
    said['bytes16'] += h.nbytes
    said['worst_abs'] = max(said['worst_abs'], float(off.max()))
    said['worst_rel'] = max(said['worst_rel'], worst_rel)
    rounded_arrays[init.name] = back
    casts.append(helper.make_node('Cast', [init.name + '__f16'], [init.name], to=TensorProto.FLOAT,
                                  name=init.name + '__f32'))
    return numpy_helper.from_array(h, init.name + '__f16')

  for k, init in enumerate(g.initializer):
    if init.data_type != TensorProto.FLOAT or init.name in graph_inputs:
      continue
    got = store(init, False)
    if got is not None:
      g.initializer[k].CopyFrom(got)
  # a Loop body's own weights move out to the graph (the body reads them as
  # outer names, as it reads any other weight), so their Casts run once and
  # not once a frame
  for body in bodies(g):
    keep = []
    for init in body.initializer:
      got = store(init, True) if init.data_type == TensorProto.FLOAT else None
      if got is None:
        keep.append(init)
      else:
        g.initializer.append(got)
        taken.add(init.name)
    if len(keep) != len(body.initializer):
      keep = [onnx.TensorProto.FromString(i.SerializeToString()) for i in keep]
      del body.initializer[:]
      body.initializer.extend(keep)
  # the Casts first: a weight is read by nodes anywhere in the graph, and by
  # the bodies of its Loops (an outer name)
  nodes = list(g.node)
  del g.node[:]
  g.node.extend(casts + nodes)
  m.metadata_props.add(key='f16_storage', value='1')
  for model in (m, r):
    # a variant names the export it was made from by its size
    # (onnx_models._variant_matches): the converted export's, when given
    for prop in model.metadata_props:
      if variant_of and prop.key == 'from_bytes':
        prop.value = str(os.path.getsize(variant_of))
  onnx.save(m, dst)
  if rounded:
    for graph in [r.graph] + list(bodies(r.graph)):
      for k, init in enumerate(graph.initializer):
        if init.name in rounded_arrays:
          graph.initializer[k].CopyFrom(numpy_helper.from_array(rounded_arrays[init.name], init.name))
    onnx.save(r, rounded)
  said.update(src_bytes=os.path.getsize(src), dst_bytes=os.path.getsize(dst), casts=len(casts))
  return said


def verify(dst):
  """The graph as written: it passes onnx's checker, every Cast added reads a
  float16 weight and writes float32, and no float16 reaches any other node."""
  import onnx
  from onnx import TensorProto
  m = onnx.load(dst)
  onnx.checker.check_model(dst)
  f16 = {i.name for i in m.graph.initializer if i.data_type == TensorProto.FLOAT16}
  read = {}
  for n in m.graph.node:
    for i in n.input:
      if i in f16:
        read.setdefault(i, []).append(n)
  bad = []
  for name in f16:
    users = read.get(name, [])
    if len(users) != 1 or users[0].op_type != 'Cast' or \
        [a.i for a in users[0].attribute if a.name == 'to'] != [TensorProto.FLOAT] or \
        users[0].output[0] != name[:-len('__f16')]:
      bad.append(name)
  return dict(f16_weights=len(f16), mistyped=bad)


# How far a float16-stored file's answer may be from its float32 original's on
# the app's fixed check input (onnx_models.check_feed, three seeds, on the
# processor): the largest difference of any float output, over that output's
# own largest value. Rounding the weights moved it 0.0028 at the median and
# 0.0092 at the most over the 383 files of the 2026-10-14 set (every agent's
# step, card variant and whole-game export, and the rating policy's two;
# logits off by up to 0.05 where they reach 10 to 25); a file whose casts or
# weights were wrong is off by the size of the output itself.
F16_REL = 0.03
F16_SEEDS = (11, 1, 2)


def _check_feeds(session, seeds):
  """onnx_models.check_feed for each seed; a pre-embedded variant gets its
  export's fixed input through its own recipe (as onnx_models.self_check
  asks it)."""
  import json
  import types
  import onnx_models
  meta = session.get_modelmeta().custom_metadata_map or {}
  if 'preembed' not in meta:
    return [onnx_models.check_feed(session, s) for s in seeds]
  pre = onnx_models.PreEmbed(json.loads(meta['preembed']))
  orig = types.SimpleNamespace(get_inputs=lambda: [
      types.SimpleNamespace(name=n, shape=s, type=t) for n, s, t in pre.recipe['source_inputs']])
  takes = {i.name for i in session.get_inputs()}
  return [{k: v for k, v in pre(onnx_models.check_feed(orig, s)).items() if k in takes}
          for s in seeds]


def drift(src, dst, seeds=F16_SEEDS, threads=4):
  """The float16 file `dst` against its float32 original `src` on the
  processor: dict(rel, worst, output, samples_differ, samples, rows_differ,
  rows). rel is what F16_REL holds.

  A step draws its presses one head at a time, each head reading the presses
  drawn before it, so a press that comes out the other way (one in a few
  thousand: the odds moved in their fourth digit) changes every later head's
  logits in that row by as much as they are big (falcon_delay_18_vs_fox:
  logits9 26 off in the one row of twelve whose press differed). So the
  floats of a row are compared only where every press of the row is the same,
  and the rows whose presses differ are counted: drift_ok holds them to a
  quarter of the rows (a file with a wrong head draws other presses in nearly
  every row)."""
  import onnxruntime as ort

  def session(p):
    so = ort.SessionOptions()
    so.log_severity_level = 3
    so.intra_op_num_threads = threads
    return ort.InferenceSession(p, so, providers=['CPUExecutionProvider'])
  a, b = session(src), session(dst)
  names = [o.name for o in a.get_outputs()]
  out = dict(rel=0.0, worst=0.0, output=None, samples_differ=0, samples=0, rows_differ=0, rows=0)
  for feed in _check_feeds(a, seeds):
    got = [(n, np.asarray(x), np.asarray(y))
           for n, x, y in zip(names, a.run(None, feed), b.run(None, feed))]
    ints = [(x, y) for _, x, y in got if x.dtype.kind != 'f' and x.ndim]
    rows = ints[0][0].shape[0] if ints else None
    same = None
    if rows is not None:
      same = np.ones(rows, bool)
      for x, y in ints:
        if x.shape[:1] == (rows,):
          same &= (x.reshape(rows, -1) == y.reshape(rows, -1)).all(axis=1)
      out['rows'] += rows
      out['rows_differ'] += int((~same).sum())
    for name, x, y in got:
      if x.dtype.kind != 'f':
        out['samples_differ'] += int((x != y).sum())
        out['samples'] += int(x.size)
        continue
      if same is not None and x.shape[:1] == (rows,):
        x, y = x[same], y[same]
      if not x.size:
        continue
      off = float(np.nanmax(np.abs(x.astype(np.float64) - y)))
      scale = float(np.nanmax(np.abs(x)))
      rel = off / scale if scale else (0.0 if off == 0 else float('inf'))
      out['worst'] = max(out['worst'], off)
      if rel > out['rel'] or (rel != rel):
        out.update(rel=rel, output=name)
  return out


def drift_ok(d):
  """A drift() result the refresh accepts."""
  return d['rel'] <= F16_REL and d['rows_differ'] * 4 <= d['rows']


def main():
  ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
  ap.add_argument('src')
  ap.add_argument('dst')
  ap.add_argument('--rounded')
  ap.add_argument('--min', type=int, default=256)
  ap.add_argument('--rel', type=float, default=0.01)
  ap.add_argument('--variant-of', help='SRC is a .pre or .lstm variant: the converted export it '
                  'now belongs to (its size is written where the variant names its export)')
  ap.add_argument('--drift', action='store_true',
                  help="also hold DST's answer to SRC's on the fixed check input (F16_REL)")
  ap.add_argument('--exact', action='store_true',
                  help='store a weight as float16 only when float16 holds it exactly')
  a = ap.parse_args()
  said = convert(a.src, a.dst, a.rounded, a.min, a.rel, a.variant_of, a.exact)
  said.update(verify(a.dst))
  bad = bool(said['mistyped'])
  if a.drift:
    said['drift'] = drift(a.src, a.dst)
    bad = bad or not drift_ok(said['drift'])
  print(os.path.basename(a.src), said)
  return 1 if bad else 0


if __name__ == '__main__':
  sys.exit(main())
