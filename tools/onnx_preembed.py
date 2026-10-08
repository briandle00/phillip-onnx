"""The agents' step graphs with their input encoding taken out, for the
graphics card.

A step export (<name>.step.onnx, tools/export_agent_onnx.py) reads ~120 game
fields one input each, and turns each into what the network reads with its
own small ops: a cast, a scale and a clip, a one-hot, an embedding row. For
'gm' that is 618 of the graph's 821 nodes, against ~200 for the network
itself; on a card every node is a kernel launch, so a step of 64 rows took
7.2 ms where its arithmetic is ~1.2 ms (RTX 4070, measured 2026-10-02).

This writes <name>.step.pre.onnx next to <name>.step.onnx. Its graph builds
the very same input vector (the one the first matrix multiply reads) in ~30
nodes from a handful of stacked inputs:

  pre<k>_f     the scaled-and-clipped floats, raw: [B, n] float32 - scaled,
               clipped by the same Mul, Max and Min the export does, with a
               vector of the export's constants instead of one node a field
  pre<k>_v     the fields that go in as they are: flags as the export's two
               values (facing is 1/-1), [B, n] float32
  pre<k>_oh    one-hot fields as rows of one block identity table: [B, n] int64
  pre<k>_emb   embedding rows (several tables stacked in one): [B, n] int64
  pre_mask<j>  a boolean the network reads as it is (the items' exists)

and puts them in the export's column order with one Gather. Everything after
that vector is the export's own nodes, untouched. onnx_models.PreEmbed turns
a step's usual feed (StepNet.run's) into those inputs; the recipe it follows
is in the variant's metadata ('preembed'), worked out here from the graph.

The vector is the same bits the export's encoding gives (a one-hot is 0 and
1 either way, an embedding row is a copy, a scale is the same fp32 multiply),
so every output is too: checked before writing, on the processor, on seeded
inputs over each field's whole range (out-of-range one-hots, negative item
types, NaN and -0.0 floats), to exactly 0. A field this does not recognize
keeps the export's own nodes, so nothing is ever guessed.

  ..\\export-env\\Scripts\\pythonw tools\\onnx_preembed.py PATH\\gm.step.onnx [...]

Needs the `onnx` package (export-env has it; the app's Python does not).
"""

import collections
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
  sys.path.insert(0, HERE)

FORMAT = 1
CHECK_ROWS = (1, 7, 64)             # batch sizes the processor check runs at
CHECK_SEEDS = 6                     # seeded feeds a size

# the integer ops an embedding's row number may be worked out with (the
# character-action table's: (character * 399 + action), in the inputs' own
# unsigned types)
PROGRAM_OPS = {'Cast', 'Mul', 'Add', 'Sub', 'GreaterOrEqual', 'Greater', 'Less', 'LessOrEqual',
               'Equal', 'And', 'Or', 'Not', 'Expand'}


def variant_path(path):
  return path[:-len('.onnx')] + '.pre.onnx'


class _Graph:
  """The export's graph, with what the analysis asks of it."""

  def __init__(self, model):
    import onnx
    from onnx import numpy_helper
    self.onnx = onnx
    g = model.graph
    self.g = g
    self.consts = {i.name: numpy_helper.to_array(i) for i in g.initializer}
    for n in g.node:
      if n.op_type == 'Constant' and n.attribute and n.attribute[0].name == 'value':
        self.consts[n.output[0]] = numpy_helper.to_array(n.attribute[0].t)
    self.prod = {o: n for n in g.node for o in n.output}
    self.cons = collections.defaultdict(list)
    for n in g.node:
      for i in n.input:
        self.cons[i].append(n)
    self.inputs = {i.name: i for i in g.input}
    self.shapes = {}
    for v in list(g.value_info) + list(g.input) + list(g.output):
      t = v.type.tensor_type
      self.shapes[v.name] = [d.dim_value if d.HasField('dim_value') else d.dim_param
                             for d in t.shape.dim]
    self.types = {i.name: i.type.tensor_type.elem_type for i in g.input}
    # the game's inputs: what onnx_models.encode gives (named '.state.p0.x')
    self.game = {n for n in self.inputs if n.startswith('.')}
    self.deps = self._deps()

  def _deps(self):
    """{tensor: the graph inputs its values depend on}; a Shape's output
    depends on nothing (only the batch size)."""
    deps = {n: {n} for n in self.inputs}
    for n in self.g.node:
      d = set()
      if n.op_type != 'Shape':
        for i in n.input:
          d |= deps.get(i, set())
      for o in n.output:
        deps[o] = d
    return deps

  def attr(self, node, name, default=None):
    for a in node.attribute:
      if a.name == name:
        return self.onnx.helper.get_attribute_value(a)
    return default

  def const(self, t):
    return self.consts.get(t)

  def scalar(self, t):
    c = self.const(t)
    if c is None or c.size != 1:
      return None
    return c.reshape(())

  def op(self, t, *types):
    n = self.prod.get(t)
    return n if n is not None and (not types or n.op_type in types) else None

  def in_region(self, t):
    d = self.deps.get(t)
    return bool(d) and d <= self.game

  def strip(self, t):
    """Past the broadcasting the export puts round a field to make it a
    column (Reshape to [B, 1], Expand to the same, or an Unsqueeze)."""
    while True:
      n = self.op(t, 'Expand', 'Reshape', 'Unsqueeze')
      if n is None:
        return t
      t = n.input[0]

  def column(self, leaf):
    """The tensor a one-wide block is a column of, or None when the block is
    not one wide or the column's shape is not the block's less its last 1."""
    t = self.strip(leaf)
    a, b = self.shapes.get(leaf), self.shapes.get(t)
    if a is not None and (not a or a[-1] != 1):
      return None
    if a is not None and b is not None and len(a) != len(b) + 1:
      return None
    return t

  def source(self, t):
    """The game inputs a tensor is, as they are: [name] for an input, the
    names in order for a stack of them (Concat on axis 1 of each Unsqueezed
    on axis 1, the items'), else None."""
    if t in self.game:
      return [t]
    n = self.op(t, 'Concat')
    if n is None or self.attr(n, 'axis') != 1:
      return None
    names = []
    for i in n.input:
      u = self.op(i, 'Unsqueeze')
      if u is None or u.input[0] not in self.game:
        return None
      axes = self.const(u.input[1]) if len(u.input) > 1 else np.array(self.attr(u, 'axes'))
      if axes is None or list(np.ravel(axes)) != [1]:
        return None
      names.append(u.input[0])
    return names

  def cast_source(self, t, to):
    """source(t), or source of a Cast to `to` (an ONNX type number)."""
    s = self.source(t)
    if s is not None:
      return s, False
    n = self.op(t, 'Cast')
    if n is not None and self.attr(n, 'to') == to:
      s = self.source(n.input[0])
      if s is not None:
        return s, True
    return None, False

  def const_value(self, t):
    """A tensor that depends on no input, worked out (a scalar): Expand of a
    constant to the batch, times a constant (the rating column: 1 x 2500)."""
    c = self.const(t)
    if c is not None:
      return c
    n = self.op(t)
    if n is None or self.deps.get(t):
      return None
    if n.op_type == 'Expand':
      return self.const_value(n.input[0])
    if n.op_type in ('Mul', 'Add', 'Sub', 'Div'):
      a, b = self.const_value(n.input[0]), self.const_value(n.input[1])
      if a is None or b is None or a.size != 1 or b.size != 1:
        return None
      f = {'Mul': np.multiply, 'Add': np.add, 'Sub': np.subtract, 'Div': np.divide}[n.op_type]
      return np.asarray(f(a.reshape(()), b.reshape(())))
    return None


FLOAT, INT64, BOOL = 1, 7, 9


def _piece(G, leaf):
  """What one block of an input concat is: a dict with 'kind' (f, v, oh, emb,
  embsum) and how to make it, or None (it keeps the export's own nodes).
  One-wide blocks (f, v) are read past the broadcasting that makes them a
  column; table blocks only as they are."""
  t = G.column(leaf)
  n = G.op(t) if t is not None else None
  # a float field: [Cast] -> Mul -> Max -> Min, all three (the network's
  # float embedding: scale, then clip)
  if n is not None and n.op_type == 'Min':
    hi = G.scalar(n.input[1])
    mx = G.op(n.input[0], 'Max')
    if hi is None or mx is None:
      return None
    lo = G.scalar(mx.input[1])
    mul = G.op(mx.input[0], 'Mul')
    if lo is None or mul is None:
      return None
    a, s = mul.input
    if G.scalar(s) is None:
      a, s = s, a
    s = G.scalar(s)
    if s is None:
      return None
    src, cast = G.cast_source(a, FLOAT)
    if src is not None:
      if not cast and any(G.types.get(x) != FLOAT for x in src):
        return None
      return dict(kind='f', src=src, s=float(s), lo=float(lo), hi=float(hi), width=1)
    c = G.const_value(a)
    if c is not None and c.size == 1:
      return dict(kind='f', src=None, value=float(np.float32(c.reshape(()))), s=float(s),
                  lo=float(lo), hi=float(hi), width=1)
    return None
  # a flag: Where(input, on, off)
  if n is not None and n.op_type == 'Where':
    src = G.source(n.input[0])
    c1, c0 = G.scalar(n.input[1]), G.scalar(n.input[2])
    if src is None or c1 is None or c0 is None or c1.dtype != np.float32:
      return None
    if any(G.types.get(x) != BOOL for x in src):
      return None
    return dict(kind='v', src=src, on=float(c1), off=float(c0), width=1)
  # a float input as it is
  if n is None and t in G.game and G.types.get(t) == FLOAT:
    return dict(kind='v', src=[t], width=1)
  n = G.op(leaf)                       # the table blocks: as they are
  if n is not None and n.op_type == 'OneHot':
    src, _ = G.cast_source(n.input[0], INT64)
    depth = G.scalar(n.input[1])
    vals = G.const(n.input[2])
    if src is None or depth is None or vals is None or list(vals.ravel()) != [0.0, 1.0]:
      return None
    if vals.dtype != np.float32 or G.attr(n, 'axis', -1) != -1:
      return None
    return dict(kind='oh', src=src, depth=int(depth), width=int(depth))
  if n is not None and n.op_type == 'Gather':
    got = _gather(G, n)
    if got is None:
      return None
    table, src = got
    return dict(kind='emb', table=table, src=src, width=int(G.const(table).shape[1]))
  # an embedding plus a masked second one (the action's own row plus its
  # character-action row where the character and action are in range)
  if n is not None and n.op_type == 'Add':
    a = G.op(n.input[0], 'Gather')
    w = G.op(n.input[1], 'Where')
    if a is None or w is None:
      return None
    got = _gather(G, a)
    b = G.op(w.input[1], 'Gather')
    zero = G.scalar(w.input[2])
    if got is None or b is None or zero is None or zero != 0 or np.signbit(zero):
      return None
    tb = b.input[0]
    T = G.const(tb)
    if T is None or T.ndim != 2 or G.attr(b, 'axis', 0) != 0:
      return None
    prog = _program(G, [b.input[1], G.strip(w.input[0])])
    if prog is None:
      return None
    return dict(kind='embsum', table=got[0], src=got[1], table_b=tb, prog=prog,
                width=int(G.const(got[0]).shape[1]))
  return None


def _gather(G, n):
  """(table, source) of Gather(table, Cast(input to int64)) on axis 0."""
  T = G.const(n.input[0])
  if T is None or T.ndim != 2 or T.dtype != np.float32 or G.attr(n, 'axis', 0) != 0:
    return None
  src, _ = G.cast_source(n.input[1], INT64)
  if src is None:
    return None
  return n.input[0], src


def _program(G, outs):
  """The integer ops from game inputs to `outs`, as a program the host runs
  in numpy (the same types, so the same wrap-around): {'ops', 'consts',
  'inputs' (sources in order), 'outs'}; None if anything is not an op
  PROGRAM_OPS has, or not a constant, or not a game input."""
  ops, consts, inputs, names = [], [], [], {}

  def ref(t):
    if t in names:
      return names[t]
    src = G.source(t)
    if src is not None:
      names[t] = '$%d' % len(inputs)
      inputs.append(src)
      return names[t]
    c = G.const(t)
    if c is not None:
      names[t] = 'c%d' % len(consts)
      consts.append(dict(value=c.ravel().tolist(), dtype=str(c.dtype), shape=list(c.shape)))
      return names[t]
    n = G.op(t)
    if n is None or n.op_type not in PROGRAM_OPS:
      raise ValueError(t)
    if n.op_type == 'Expand':            # a constant to the batch: broadcasting does it
      if G.deps.get(n.input[0]):
        raise ValueError(t)
      names[t] = ref(n.input[0])
      return names[t]
    args = [ref(i) for i in n.input]
    out = 't%d' % len(ops)
    ops.append(dict(op=n.op_type, args=args, to=G.attr(n, 'to')))
    names[t] = out
    return out

  try:
    got = [ref(t) for t in outs]
  except ValueError:
    return None
  return dict(ops=ops, consts=consts, inputs=inputs, outs=got)


def _leaves(G, t, axis):
  """The blocks of a concat tree (Concats on `axis` inside one another)."""
  n = G.op(t, 'Concat')
  if n is None or G.attr(n, 'axis') != axis:
    return [t]
  out = []
  for i in n.input:
    out += _leaves(G, i, axis)
  return out


def _width(G, t, rank):
  s = G.shapes.get(t)
  if not s or len(s) != rank or not isinstance(s[-1], int) or not s[-1]:
    raise ValueError('%s: its width is not known' % t)
  return s[-1]


class _Builder:
  """New nodes, initializers and inputs, named pre*."""

  def __init__(self, G):
    from onnx import helper, numpy_helper
    self.G, self.h, self.nh = G, helper, numpy_helper
    self.nodes, self.inits, self.inputs = [], [], []
    self.k = 0

  def name(self, base):
    self.k += 1
    return 'pre_%s_%d' % (base, self.k)

  def init(self, base, a):
    n = self.name(base)
    self.inits.append(self.nh.from_array(np.ascontiguousarray(a), n))
    return n

  def node(self, op, ins, out=None, **attrs):
    out = out or self.name(op.lower())
    self.nodes.append(self.h.make_node(op, ins, [out], name=self.name('n' + op), **attrs))
    return out

  def input(self, name, elem, shape):
    self.inputs.append(self.h.make_tensor_value_info(name, elem, shape))
    return name


def _concat(G, B, t, axis, k, recipe):
  """Rebuild the concat tensor t (rank axis + 1): its recognized blocks from
  pre<k>_* inputs, the rest as the export computes them, one Gather into its
  column order. Fills recipe[k]; returns the blocks' pieces."""
  rank = axis + 1
  leaves = _leaves(G, t, axis)
  pieces = [_piece(G, x) for x in leaves]
  n = None
  for x, p in zip(leaves, pieces):
    if p is None:
      continue
    srcs = [p['src']] if p.get('src') is not None else []
    if p['kind'] == 'embsum':
      srcs += p['prog']['inputs']
    for s in srcs:
      want = 1 if rank == 2 else len(s)
      if (rank == 2) != (len(s) == 1) or (n is not None and want != n):
        raise ValueError('%s: blocks of different shapes' % t)
      n = want
  # a table width other than the most common one stays the export's
  widths = collections.Counter(p['width'] for p in pieces if p and p['kind'] in ('emb', 'embsum'))
  ew = widths.most_common(1)[0][0] if widths else None
  for i, p in enumerate(pieces):
    if p and p['kind'] in ('emb', 'embsum') and p['width'] != ew:
      pieces[i] = None
    elif p and p['kind'] == 'embsum' and G.const(p['table_b']).shape[1] != ew:
      pieces[i] = None
  lead = ['B'] if rank == 2 else ['B', n]
  parts, cols = [], {}                  # cols[leaf index] = first column in the buffer
  at = 0
  rec = dict(rank=rank, n=n)

  def place(idx, width):
    nonlocal at
    cols[idx] = at
    at += width

  # floats to scale and clip
  fs = [i for i, p in enumerate(pieces) if p and p['kind'] == 'f']
  if fs:
    raw = B.input('pre%d_f' % k, FLOAT, lead + [len(fs)])
    vec = lambda key: B.init('f', np.array([pieces[i][key] for i in fs], np.float32))  # noqa: E731
    y = B.node('Mul', [raw, vec('s')])
    y = B.node('Max', [y, vec('lo')])
    parts.append(B.node('Min', [y, vec('hi')]))
    for i in fs:
      place(i, 1)
    rec['f'] = [dict(src=pieces[i]['src'], value=pieces[i].get('value')) for i in fs]
  # values as they are
  vs = [i for i, p in enumerate(pieces) if p and p['kind'] == 'v']
  if vs:
    parts.append(B.input('pre%d_v' % k, FLOAT, lead + [len(vs)]))
    for i in vs:
      place(i, 1)
    rec['v'] = [dict(src=pieces[i]['src'], on=pieces[i].get('on'), off=pieces[i].get('off'))
                for i in vs]
  # one-hots: rows of a block identity table, summed (one 1 a block)
  ohs = [i for i, p in enumerate(pieces) if p and p['kind'] == 'oh']
  if ohs:
    total = sum(pieces[i]['depth'] for i in ohs)
    T = np.zeros([total + 1, total], np.float32)
    off, rows = 0, []
    for i in ohs:
      d = pieces[i]['depth']
      T[np.arange(off, off + d), np.arange(off, off + d)] = 1
      rows.append(dict(src=pieces[i]['src'], depth=d, offset=off))
      place(i, d)
      off += d
    idx = B.input('pre%d_oh' % k, INT64, lead + [len(ohs)])
    g = B.node('Gather', [B.init('oh', T), idx], axis=0)
    parts.append(B.node('ReduceSum', [g, B.init('ax', np.array([axis], np.int64))], keepdims=0))
    rec['oh'] = dict(rows=rows, zero=total)
  # embeddings: one stacked table; a sum's two rows added
  plain = [i for i, p in enumerate(pieces) if p and p['kind'] == 'emb']
  sums = [i for i, p in enumerate(pieces) if p and p['kind'] == 'embsum']
  if plain or sums:
    tables, at_row = {}, 0
    for i in plain + sums:
      for key in ('table', 'table_b'):
        tn = pieces[i].get(key)
        if tn and tn not in tables:
          tables[tn] = at_row
          at_row += G.const(tn).shape[0]
    T = np.concatenate([G.const(tn) for tn in tables] + [np.zeros([1, ew], np.float32)])
    zero = at_row
    idx = B.input('pre%d_emb' % k, INT64, lead + [len(plain) + 2 * len(sums)])
    g = B.node('Gather', [B.init('emb', T), idx], axis=0)
    if sums:
      sp = [B.name('split') for _ in range(3)] if plain else [None] + [B.name('split') for _ in range(2)]
      sizes = ([len(plain)] if plain else []) + [len(sums), len(sums)]
      outs = [s for s in sp if s]
      B.nodes.append(B.h.make_node('Split', [g, B.init('sizes', np.array(sizes, np.int64))], outs,
                                   name=B.name('nSplit'), axis=axis))
      both = B.node('Add', outs[-2:])
      g = B.node('Concat', [outs[0], both], axis=axis) if plain else both
    m = len(plain) + len(sums)
    shape = [-1, m * ew] if rank == 2 else [-1, n, m * ew]
    parts.append(B.node('Reshape', [g, B.init('shape', np.array(shape, np.int64))]))
    for i in plain + sums:
      place(i, ew)
    rec['emb'] = dict(
        plain=[dict(src=pieces[i]['src'], offset=tables[pieces[i]['table']],
                    rows=int(G.const(pieces[i]['table']).shape[0])) for i in plain],
        sums=[dict(src=pieces[i]['src'], offset=tables[pieces[i]['table']],
                   rows=int(G.const(pieces[i]['table']).shape[0]),
                   offset_b=tables[pieces[i]['table_b']],
                   rows_b=int(G.const(pieces[i]['table_b']).shape[0]), prog=pieces[i]['prog'])
              for i in sums],
        zero=zero)
  # the rest: the export's own tensors
  for i, p in enumerate(pieces):
    if p is None:
      parts.append(leaves[i])
      place(i, _width(G, leaves[i], rank))
  perm = np.concatenate([np.arange(cols[i], cols[i] + (pieces[i]['width'] if pieces[i] else
                                                        _width(G, leaves[i], rank)))
                         for i in range(len(leaves))]).astype(np.int64)
  buf = B.node('Concat', parts, axis=axis)
  B.node('Gather', [buf, B.init('perm', perm)], out=t, axis=axis)
  recipe.append(rec)
  starts = {}
  for i, p in enumerate(pieces):
    if p is not None:
      starts[i] = int(np.nonzero(perm == cols[i])[0][0])
  return leaves, pieces, starts


def _same_piece(p, q):
  keys = ('kind', 'src', 'depth', 'table', 's', 'lo', 'hi', 'on', 'off')
  return p is not None and q is not None and all(p.get(x) == q.get(x) for x in keys)


def build(src):
  """(the variant ModelProto, its recipe) for an export."""
  import onnx
  m = onnx.load(src)
  m = onnx.shape_inference.infer_shapes(m)
  G = _Graph(m)
  g = m.graph
  # the network's input vector: a Concat a Gemm reads, made of game inputs only
  X = None
  for n in g.node:
    if n.op_type in ('Gemm', 'MatMul') and G.op(n.input[0], 'Concat') and G.in_region(n.input[0]):
      X = n.input[0]
      break
  if X is None:
    raise ValueError('%s: no input vector made of the game inputs' % os.path.basename(src))
  # the export's own inputs (onnx_models.self_check asks the variant its
  # export's fixed input, through PreEmbed)
  tnames = {1: 'tensor(float)', 2: 'tensor(uint8)', 3: 'tensor(int8)', 4: 'tensor(uint16)',
            5: 'tensor(int16)', 6: 'tensor(int32)', 7: 'tensor(int64)', 9: 'tensor(bool)'}
  source_inputs = [[i.name, G.shapes.get(i.name, []), tnames.get(G.types[i.name], 'tensor(float)')]
                   for i in g.input]
  B = _Builder(G)
  recipe, masks, rebuilt = [], [], {X}
  leaves, pieces, starts = _concat(G, B, X, 1, 0, recipe)
  # what the network reads besides the vector (the controller head's last
  # press): a block of the vector, sliced out of it
  slices = {}
  for n in g.node:
    if G.in_region(n.output[0]) or n.op_type == 'Shape':
      continue
    for i in n.input:
      if i in rebuilt or i in slices or not G.in_region(i):
        continue
      p = _piece(G, i)
      for j, q in enumerate(pieces):
        if _same_piece(p, q):
          a = starts[j]
          B.node('Slice', [X, B.init('s', np.array([a], np.int64)),
                           B.init('e', np.array([a + q['width']], np.int64)),
                           B.init('a', np.array([1], np.int64))], out=i)
          slices[i] = True
          break
  # inside a block the export still computes (the items' network): a concat
  # of stacked fields is rebuilt the same way, a mask is read as it is
  k = 1
  todo = [x for x, p in zip(leaves, pieces) if p is None]
  seen = set()
  while todo:
    t = todo.pop()
    if t in seen or t in G.game:
      continue
    seen.add(t)
    n = G.op(t)
    if n is None:
      continue
    for j, i in enumerate(n.input):
      if not G.in_region(i) or i in rebuilt:
        continue
      c = G.op(i, 'Concat')
      if c is not None and G.attr(c, 'axis') == 2 and len(G.shapes.get(i, [])) == 3:
        try:
          _concat(G, B, i, 2, k, recipe)
          rebuilt.add(i)
          k += 1
          continue
        except ValueError:
          pass
      if n.op_type == 'Where' and j == 0:
        s = G.source(G.strip(i))
        if s is not None and all(G.types.get(x) == BOOL for x in s):
          name = 'pre_mask%d' % len(masks)
          lead = ['B', 1] if len(s) == 1 else ['B', len(s), 1]
          B.input(name, BOOL, lead)
          masks.append(dict(name=name, src=s))
          n.input[j] = name
          continue
      todo.append(i)
  # the new graph: the export's nodes but those whose outputs are rebuilt (in
  # a topological order: the new ones read inputs, kept tensors and each other)
  made = {o for n in B.nodes for o in n.output}
  nodes = [n for n in g.node if not any(o in made for o in n.output)]
  nodes = _toposort(B.nodes + nodes, {i.name for i in g.input} | {i.name for i in B.inputs}
                    | set(G.consts) | {x.name for x in B.inits})
  g.ClearField('node')
  g.node.extend(nodes)
  g.initializer.extend(B.inits)
  g.input.extend(B.inputs)
  _prune(g)
  _simplify(m)
  _prune(g)
  rec = dict(format=FORMAT, concats=recipe, masks=masks,
             inputs=[i.name for i in g.input], source_inputs=source_inputs)
  return m, rec


def _prune(g):
  """Drop what no longer feeds an output: nodes, initializers, inputs, and
  the shapes of tensors that are gone."""
  outs = {o.name for o in g.output}
  nodes = list(g.node)
  while True:
    used = {i for n in nodes for i in n.input} | outs
    kept = [n for n in nodes if any(o in used for o in n.output)]
    if len(kept) == len(nodes):
      break
    nodes = kept
  have = {o for n in nodes for o in n.output}
  inits = [i for i in g.initializer if i.name in used]
  inputs = [i for i in g.input if i.name in used]
  info = [v for v in g.value_info if v.name in have]
  for field, keep in (('node', nodes), ('initializer', inits), ('input', inputs),
                      ('value_info', info)):
    g.ClearField(field)
    getattr(g, field).extend(keep)


# ops that broadcast their inputs elementwise: an input Expanded to a shape
# the others already give is the same values without the Expand
BROADCAST_OPS = {'Add', 'Sub', 'Mul', 'Div', 'Where', 'Max', 'Min', 'And', 'Or', 'Equal',
                 'Less', 'Greater', 'LessOrEqual', 'GreaterOrEqual', 'Pow'}


def _broadcast(shapes):
  rank = max(len(s) for s in shapes)
  out = []
  for k in range(rank):
    dims = {s[len(s) - rank + k] for s in shapes if len(s) - rank + k >= 0} - {1}
    if len(dims) > 1:
      return None
    out.append(dims.pop() if dims else 1)
  return out


def _simplify(m):
  """The export's broadcasting plumbing out of what is left (the reset's
  Where on each state, the LSTM biases, the items' reshapes): a Reshape to
  a shape worked out from the batch size gets that shape as a constant (-1
  for the batch), and an Expand into an elementwise op whose other inputs
  already give the output's shape goes. The same values either way; the
  processor check after says so."""
  import onnx
  from onnx import numpy_helper
  g = m.graph
  inferred = onnx.shape_inference.infer_shapes(m)
  shapes = {}
  for v in list(inferred.graph.value_info) + list(g.input) + list(g.output):
    t = v.type.tensor_type
    if t.HasField('shape'):
      shapes[v.name] = [d.dim_value if d.HasField('dim_value') else (d.dim_param or None)
                        for d in t.shape.dim]
  for i in g.initializer:
    shapes[i.name] = list(i.dims)
  known = lambda s: s is not None and all(d is not None for d in s)  # noqa: E731
  inits = {i.name for i in g.initializer}
  prod = {o: n for n in g.node for o in n.output}
  added = []
  for n in g.node:
    if n.op_type == 'Reshape' and n.input[1] not in inits:
      s = shapes.get(n.output[0])
      if known(s) and sum(not isinstance(d, int) for d in s) == 1 and \
          all(d > 0 for d in s if isinstance(d, int)):
        name = 'pre_rs_%s' % n.output[0]
        added.append(numpy_helper.from_array(
            np.array([d if isinstance(d, int) else -1 for d in s], np.int64), name))
        n.input[1] = name
  for n in g.node:
    if n.op_type not in BROADCAST_OPS:
      continue
    want = shapes.get(n.output[0])
    if not known(want):
      continue
    for j, i in enumerate(n.input):
      e = prod.get(i)
      if e is None or e.op_type != 'Expand' or not known(shapes.get(e.input[0])):
        continue
      trial = [shapes.get(x) for x in n.input]
      trial[j] = shapes[e.input[0]]
      if all(known(t) for t in trial) and _broadcast(trial) == want:
        n.input[j] = e.input[0]
  g.initializer.extend(added)


def _toposort(nodes, have):
  have = set(have) | {''}
  out, left = [], list(nodes)
  while left:
    rest = []
    for n in left:
      if all(i in have for i in n.input):
        out.append(n)
        have.update(n.output)
      else:
        rest.append(n)
    if len(rest) == len(left):
      raise ValueError('the new graph has a cycle or a missing input: %s'
                       % sorted({i for n in rest for i in n.input if i not in have})[:5])
    left = rest
  return out


def to_preembed(src, dst=None, feeds=None):
  """Write src's pre-embedded variant; returns its path. feeds: more step
  feeds (a list of {input: array} of the export's own inputs) to check on."""
  import onnx
  dst = dst or variant_path(src)
  m, rec = build(src)
  with open(src + '.json', encoding='utf-8') as f:
    exported = json.load(f).get('exported')
  for k, v in (('from_exported', str(exported)), ('from_bytes', str(os.path.getsize(src))),
               ('preembed', json.dumps(rec))):
    p = m.metadata_props.add()
    p.key, p.value = k, v
  onnx.checker.check_model(m)
  tmp = dst + '.tmp'
  onnx.save(m, tmp)
  try:
    worst, runs = compare(src, tmp, feeds)
  except Exception:
    os.remove(tmp)
    raise
  if worst != 0:
    os.remove(tmp)
    raise ValueError('the variant is %.3g off the original on the processor' % worst)
  os.replace(tmp, dst)
  n0 = len(onnx.load(src, load_external_data=False).graph.node)
  print('%s: %d nodes (the export: %d), the same bits on the processor on %d feeds'
        % (os.path.basename(dst), len(m.graph.node), n0, runs))
  return dst


def check_feeds(session, meta, rows, seed):
  """Seeded step feeds over each field's whole range: one-hots past their
  depth and below zero, gathers within their table (beyond it the export
  itself fails), floats with -0.0, NaN, infinities and big values."""
  rng = np.random.default_rng(seed)
  sizes = {}

  def walk(spec, prefix):
    if spec['op'] == 'struct':
      for k, sub in spec['fields']:
        walk(sub, prefix + '.' + k)
    elif spec['op'] == 'custom_v1':
      walk(spec['mid'], prefix)
    elif spec['op'] == 'onehot':
      sizes[prefix] = spec['size']
  walk(meta['encode'], '')
  types = {'tensor(float)': np.float32, 'tensor(uint8)': np.uint8, 'tensor(uint16)': np.uint16,
           'tensor(int8)': np.int8, 'tensor(int16)': np.int16, 'tensor(int32)': np.int32,
           'tensor(int64)': np.int64, 'tensor(bool)': np.bool_}
  feed = {}
  for i in session.get_inputs():
    shape = [d if isinstance(d, int) else rows for d in i.shape]
    t = types.get(i.type, np.float32)
    if i.name == 'temperature':
      a = np.asarray(rng.choice([1.0, 0.7]), np.float32)
    elif i.name.startswith('noise'):
      a = rng.random(shape).astype(np.float32).clip(1e-6, 1 - 1e-6)
    elif t == np.float32:
      a = (rng.normal(0, 1, shape) * rng.choice([1, 30, 300], shape)).astype(np.float32)
      special = np.array([0.0, -0.0, np.nan, np.inf, -np.inf, 1e30], np.float32)
      pick = rng.random(shape) < (0.05 if i.name.startswith('.') else 0)
      a[pick] = rng.choice(special, int(pick.sum()))
    elif t == np.bool_:
      a = rng.random(shape) < (0.02 if i.name == 'reset' else 0.5)
    else:
      size = sizes.get(i.name, 64)
      # what picks an embedding row stays within its table (encode clamps
      # the action and refuses a character past the list)
      inside = i.name.startswith('.state') and i.name.endswith(('.action', '.character'))
      a = rng.integers(0, size, shape).astype(t)
      if not inside:                     # some out of range, both ways where the type has them
        lo = 0 if np.iinfo(t).min == 0 else -size - 3
        wild = rng.random(shape) < 0.3
        a[wild] = rng.integers(lo, size + 3, int(wild.sum())).astype(t)
    feed[i.name] = a
  return feed


def compare(src, variant, feeds=None):
  """(the largest output difference, the number of feeds): the original and
  the variant on the processor, the variant fed by onnx_models.PreEmbed."""
  import onnxruntime as ort
  import onnx_models
  so = ort.SessionOptions()
  so.log_severity_level = 3
  so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
  # a few threads: the check runs beside whatever the PC is doing (a bulk
  # rebuild converts ~110 agents one after the other)
  so.intra_op_num_threads = int(os.environ.get('PHILLIP_ONNX_PREEMBED_THREADS') or 4)
  a = ort.InferenceSession(src, so, providers=['CPUExecutionProvider'])
  b = ort.InferenceSession(variant, so, providers=['CPUExecutionProvider'])
  pre = onnx_models.PreEmbed(json.loads(b.get_modelmeta().custom_metadata_map['preembed']))
  meta = onnx_models.read_meta(src)
  takes = {i.name for i in b.get_inputs()}
  names = [o.name for o in a.get_outputs()]
  all_feeds = [check_feeds(a, meta, rows, seed) for rows in CHECK_ROWS
               for seed in range(CHECK_SEEDS)] + list(feeds or [])
  worst = 0.0
  for feed in all_feeds:
    want = dict(zip(names, a.run(None, feed)))
    mine = {k: v for k, v in pre(feed).items() if k in takes}
    got = dict(zip([o.name for o in b.get_outputs()], b.run(None, mine)))
    for k, w in want.items():
      g = np.asarray(got[k])
      if g.dtype != w.dtype or g.shape != w.shape:
        return float('inf'), len(all_feeds)
      if not np.array_equal(g, w, equal_nan=w.dtype.kind == 'f'):
        if w.dtype.kind == 'f':
          d = np.abs(g.astype(np.float64) - w)
          worst = max(worst, float(np.nanmax(d)) if np.isfinite(d).any() else float('inf'), 1e-30)
        else:
          worst = max(worst, float(np.abs(g.astype(np.float64) - w).max()))
  return worst, len(all_feeds)


def main(argv=None):
  argv = sys.argv[1:] if argv is None else argv
  if not argv:
    print(__doc__)
    return 2
  for src in argv:
    to_preembed(src)
  return 0


if __name__ == '__main__':
  sys.exit(main())
