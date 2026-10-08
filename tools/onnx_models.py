"""The value function and the rating policy on onnxruntime.

Why this exists: the two checkpoints are JAX models, and JAX + flax + the
ratings-embed checkout is a second environment of its own (~700 MB) that no
installer should have to carry. onnxruntime is ~40 MB, runs in the app's own
Python, and reads the same numbers to within a few millionths - measured on
real replays by tools/export_onnx.py and tests/test_onnx.py.

tools/export_onnx.py writes two files per checkpoint, next to it:

  <name>.onnx        the network, weights included
  <name>.onnx.json   what it was exported from, and how a replay becomes its
                     inputs (the "encode spec", below)

The encode spec: slippi-ai turns a parsed replay into the integer and float
arrays the network reads with `network.encode()`. That is plain numpy, but it
lives in a module that imports jax, optax and tensorflow-probability at the
top - so the exporter walks the embedding once and writes down what each leaf
gets (a cast, a clip, a bucket), and encode() below follows that recipe. The
exporter checks the recipe against the real encode on real replays before it
writes anything, and the tests do it again.

Nothing here imports jax or tensorflow.
"""

import collections
import dataclasses
import importlib.util
import json
import os
import typing

import numpy as np

FORMAT = 1                      # bumped when the .json layout changes


def available():
  """onnxruntime can be imported here (not whether any model is exported)."""
  return importlib.util.find_spec('onnxruntime') is not None


def meta_path(onnx_path):
  return onnx_path + '.json'


def read_meta(path):
  """An export's .json on its own - everything about the model except its
  weights. A process that only encodes inputs and decodes outputs (a batching
  client, whose inference happens elsewhere) reads this and never opens the
  .onnx itself."""
  with open(meta_path(path), encoding='utf-8') as f:
    meta = json.load(f)
  if meta.get('format') != FORMAT:
    raise ValueError('%s was exported by a different version of '
                     'tools/export_onnx.py - export it again' % os.path.basename(path))
  return meta


def observation_config(meta):
  from slippi_ai import observations
  return dataclass_from_dict(observations.ObservationConfig, meta['observation'])


# --- the encode spec ------------------------------------------------------------

def _get(value, key):
  return value[key] if isinstance(value, dict) else getattr(value, key)


def encode(spec, value, prefix=''):
  """Follow an encode spec over a parsed StateAction (or any part of one).

  Returns {name: array}, one entry per leaf the network reads, named the way
  the exporter named the model's inputs ('.state.p0.x', '.action.buttons').
  Fields the network does not read are left out, as slippi-ai's encode leaves
  them empty."""
  op = spec['op']
  if op == 'struct':
    out = {}
    for key, sub in spec['fields']:
      out.update(encode(sub, _get(value, key), prefix + '.' + key))
    return out
  if op == 'custom_v1':
    # the controller becomes (buttons, main stick) bucket labels, with
    # slippi-ai's own bucketer - plain numpy, importable without jax
    from slippi_ai.action_space import custom_v1 as cv1
    bucketer = dataclass_from_dict(cv1.Config, spec['config']).create_bucketer()
    return encode(spec['mid'], bucketer.bucket(value), prefix)
  if op not in ('cast', 'discrete', 'onehot'):
    raise ValueError('unknown encode op %r at %s' % (op, prefix or '(top)'))
  x = np.asarray(value)
  dtype = np.dtype(spec['dtype'])
  if op == 'cast':
    return {prefix: x.astype(dtype, copy=False)}
  if op == 'discrete':
    if x.dtype != np.float32:
      raise ValueError('%s: expected float32, got %s' % (prefix, x.dtype))
    return {prefix: (x * spec['n'] + 0.5).astype(dtype)}
  # onehot
  size, policy = spec['size'], spec['policy']
  if policy == 'CLAMP':
    x = np.clip(x, 0, size - 1)
  elif policy == 'ERROR':
    if np.any(x < 0):
      raise ValueError('Got negative input in %s' % prefix)
    if np.any(x >= size):
      raise ValueError('Got invalid input %s >= %d in %s' % (np.max(x), size, prefix))
  elif policy == 'EXTRA':
    bad = (x < 0) | (x >= size)
    if np.any(bad):
      x = x.copy()
      x[bad] = size
  # EMPTY: out-of-range values are left for the graph's one-hot to zero
  return {prefix: x.astype(dtype, copy=False)}


def dataclass_from_dict(cls, d):
  """slippi-ai's configs are nested dataclasses saved as dicts. Its own
  helper lives in a module that needs absl and fancyflags; this is the part
  of it the configs here use."""
  if not dataclasses.is_dataclass(cls) or not isinstance(d, dict):
    return d
  hints = typing.get_type_hints(cls)
  kwargs = {}
  for f in dataclasses.fields(cls):
    if f.name in d:
      kwargs[f.name] = dataclass_from_dict(hints.get(f.name, f.type), d[f.name])
  return cls(**kwargs)


# --- the models -----------------------------------------------------------------

def card_provider():
  """Which graphics card provider this onnxruntime has, in the order the app
  takes them: 'cuda' (onnxruntime-gpu: NVIDIA), then 'dml' (onnxruntime-
  directml: DirectML, any DirectX 12 card - AMD, Intel, NVIDIA), else None.
  It says nothing about whether the card's libraries load: a session finds
  that out, and says so."""
  try:
    import onnxruntime as ort
  except ImportError:
    return None
  have = ort.get_available_providers()
  if 'CUDAExecutionProvider' in have:
    return 'cuda'
  if 'DmlExecutionProvider' in have:
    return 'dml'
  return None


def gpu_available():
  """onnxruntime here can run on a graphics card (card_provider)."""
  return card_provider() is not None


def dml_device():
  """The DirectML adapter to use (PHILLIP_ONNX_DML_DEVICE, else 0: Windows' own
  order of DirectX 12 adapters, the primary first)."""
  try:
    return int(os.environ.get('PHILLIP_ONNX_DML_DEVICE') or 0)
  except ValueError:
    return 0


def cuda_options(device_id=0, mem_limit=None):
  """The CUDA provider's options, with TF32 OFF. onnxruntime turns it on by
  default (`use_tf32` reads 1 on a fresh session): matmuls then round their
  inputs to 10 bits of mantissa, and a [4096, 1536] x [1536, 6144] product
  came out 3e-4 off in relative terms, against 2e-6 with it off - measured on
  this machine's RTX 4070. The rating policy's check is 1e-4 on the per-frame
  nll, so TF32 is not allowed anywhere near it.

  kSameAsRequested: the arena grows by what is asked for, not by doubling -
  on a shared 12 GB card the doubling is what runs out first."""
  opts = {'device_id': str(int(device_id)), 'use_tf32': '0',
          'arena_extend_strategy': 'kSameAsRequested'}
  if mem_limit:
    opts['gpu_mem_limit'] = str(int(mem_limit))
  return opts


_dlls = []


def _preload_cuda():
  """The CUDA and cuDNN DLLs from the nvidia-* wheels beside onnxruntime-gpu
  (no system CUDA install), loaded once, before the first CUDA session -
  otherwise Windows looks on PATH, where an old toolkit may be first."""
  if _dlls:
    return
  import onnxruntime as ort
  if hasattr(ort, 'preload_dlls'):
    ort.preload_dlls()
  _dlls.append(True)


def card_device(device):
  """A device as a session takes it: 'cuda' means the graphics card, whichever
  provider this onnxruntime has for it (card_provider): 'cuda[:N]' with
  onnxruntime-gpu, 'dml:N' with onnxruntime-directml. 'dml[:N]' and 'cpu'
  are themselves."""
  kind = device.split(':')[0]
  if kind != 'cuda':
    return device
  got = card_provider()
  if got == 'dml':
    return 'dml:%d' % (int(device.split(':')[1]) if ':' in device else dml_device())
  return device


def _providers(device, mem_limit=None):
  """onnxruntime's providers list for 'cpu', 'cuda[:N]' or 'dml[:N]'."""
  if device == 'cpu':
    return ['CPUExecutionProvider']
  kind = device.split(':')[0]
  dev = int(device.split(':')[1]) if ':' in device else 0
  import onnxruntime as ort
  have = ort.get_available_providers()
  if kind == 'dml':
    if 'DmlExecutionProvider' not in have:
      raise RuntimeError('this onnxruntime has no DirectML provider (it is %s); the card '
                         'needs onnxruntime-directml' % ort.__version__)
    return [('DmlExecutionProvider', {'device_id': dev}), 'CPUExecutionProvider']
  if kind != 'cuda':
    raise ValueError('unknown device %r' % device)
  if 'CUDAExecutionProvider' not in have:
    raise RuntimeError('this onnxruntime has no CUDA provider (it is %s); the GPU '
                       'needs onnxruntime-gpu' % ort.__version__)
  _preload_cuda()
  return [('CUDAExecutionProvider', cuda_options(dev, mem_limit)), 'CPUExecutionProvider']


CHECK_TOL = 1e-3                       # a card's answer this close to the processor's is right
_CPU_ANSWERS = {}                       # path -> (feed, outputs) on the processor


def check_feed(session, seed=11):
  """A small fixed input for a session: every symbolic dimension 4 (the
  value function's time 6), floats from a seeded normal (noise inputs
  uniform, temperature 1), integers and flags zero."""
  rng = np.random.default_rng(seed)
  types = {'tensor(float)': np.float32, 'tensor(uint8)': np.uint8, 'tensor(uint16)': np.uint16,
           'tensor(int8)': np.int8, 'tensor(int16)': np.int16, 'tensor(int32)': np.int32,
           'tensor(int64)': np.int64, 'tensor(bool)': np.bool_}
  feed = {}
  for i in session.get_inputs():
    shape = [d if isinstance(d, int) else (6 if d == 'T' else 4) for d in i.shape]
    t = types.get(i.type, np.float32)
    if t == np.float32:
      if i.name == 'temperature':
        a = np.asarray(1.0, np.float32)
      elif i.name.startswith('noise'):
        a = rng.random(shape).astype(np.float32).clip(1e-6, 1 - 1e-6)
      else:
        a = rng.normal(0, 0.5, shape).astype(np.float32)
    else:
      a = np.zeros(shape, t)
    feed[i.name] = a
  return feed


def self_check(path, session, source=None, prepare=None):
  """How far a session's floating-point outputs are from the processor's on
  the same small fixed input (check_feed): the largest absolute difference.
  The processor's answer is worked out once a process per model.

  source, prepare: a pre-embedded variant is asked its export's fixed input
  (check_feed of `source`) through `prepare` (its PreEmbed) - the same game
  its export is checked on, not random values where flags and table rows go
  (falco_d18_vs_fox_v4's variant was 0.0011 off on those, its export 0.0003
  off on its own, and the two the same bits on any real input)."""
  import onnxruntime as ort
  if path not in _CPU_ANSWERS:
    so = ort.SessionOptions()
    so.log_severity_level = 3
    cpu = ort.InferenceSession(path, so, providers=['CPUExecutionProvider'])
    if source is None:
      feed = check_feed(cpu)
    else:
      # the export's inputs as the variant's recipe lists them (else the
      # export itself, read for its inputs)
      sig = getattr(prepare, 'recipe', {}).get('source_inputs')
      if sig:
        import types
        orig = types.SimpleNamespace(get_inputs=lambda: [
            types.SimpleNamespace(name=n, shape=s, type=t) for n, s, t in sig])
      else:
        orig = ort.InferenceSession(source, so, providers=['CPUExecutionProvider'])
      takes = {i.name for i in cpu.get_inputs()}
      feed = {k: v for k, v in prepare(check_feed(orig)).items() if k in takes}
      del orig
    names = [o.name for o in cpu.get_outputs()]
    _CPU_ANSWERS[path] = (feed, dict(zip(names, cpu.run(None, feed))))
  feed, want = _CPU_ANSWERS[path]
  names = [o.name for o in session.get_outputs()]
  got = dict(zip(names, session.run(None, feed)))
  worst = 0.0
  for name, ref in want.items():
    ref = np.asarray(ref)
    if ref.dtype.kind != 'f':
      continue                          # samples and flags follow the floats
    out = np.asarray(got[name], np.float64)
    worst = max(worst, float(np.max(np.abs(out - ref.astype(np.float64)))) if ref.size else 0.0)
  return worst


def lstm_variant(path):
  """<name>.lstm.onnx beside <name>.onnx (tools/onnx_lstm.py: the Loops as
  ONNX's LSTM op, one cuDNN call a sequence on a card), when it is there and
  was made from this export; else None."""
  v = path[:-len('.onnx')] + '.lstm.onnx' if path.endswith('.onnx') else None
  return v if v and os.path.exists(v) else None


def preembed_variant(path):
  """<name>.pre.onnx beside <name>.onnx (tools/onnx_preembed.py: a step graph
  with its input encoding taken out, ~200 nodes a step instead of ~820), when
  it is there; else None. PHILLIP_ONNX_PREEMBED=0: never."""
  if os.environ.get('PHILLIP_ONNX_PREEMBED') == '0':
    return None
  v = path[:-len('.onnx')] + '.pre.onnx' if path.endswith('.onnx') else None
  return v if v and os.path.exists(v) else None


# ONNX's tensor types as numpy's (the ones an encoding's integer ops use)
_NP_TYPES = {1: np.float32, 2: np.uint8, 3: np.int8, 4: np.uint16, 5: np.int16, 6: np.int32,
             7: np.int64, 9: np.bool_, 11: np.float64, 12: np.uint32, 13: np.uint64}
_PROGRAM = {
    'Cast': lambda x, o: x[0].astype(_NP_TYPES[o['to']]),
    'Mul': lambda x, o: np.multiply(x[0], x[1]),
    'Add': lambda x, o: np.add(x[0], x[1]),
    'Sub': lambda x, o: np.subtract(x[0], x[1]),
    'GreaterOrEqual': lambda x, o: x[0] >= x[1],
    'Greater': lambda x, o: x[0] > x[1],
    'Less': lambda x, o: x[0] < x[1],
    'LessOrEqual': lambda x, o: x[0] <= x[1],
    'Equal': lambda x, o: x[0] == x[1],
    'And': lambda x, o: np.logical_and(x[0], x[1]),
    'Or': lambda x, o: np.logical_or(x[0], x[1]),
    'Not': lambda x, o: np.logical_not(x[0]),
}


def _run_program(prog, args):
  """An embedding row's integer ops (tools/onnx_preembed._program) in numpy,
  in the graph's own types: a uint8 product wraps as it does there."""
  env = {'$%d' % j: a for j, a in enumerate(args)}
  for j, c in enumerate(prog['consts']):
    env['c%d' % j] = np.array(c['value'], c['dtype']).reshape(c['shape'])
  for j, o in enumerate(prog['ops']):
    env['t%d' % j] = _PROGRAM[o['op']]([env[a] for a in o['args']], o)
  return [env[o] for o in prog['outs']]


class PreEmbed:
  """A step's usual feed (the encoded game fields, StepNet.run's) as the
  pre-embedded variant's inputs, following the recipe tools/onnx_preembed.py
  keeps in the variant: the fields stacked, the flags as their two values,
  one-hots and embeddings as table rows. Every other input passes through.

  Rows are worked out the way the export's ops would: a one-hot index past
  its depth is the zero row (negative ones count from the end, as OneHot
  does); an embedding index outside its table is an error, as the export's
  Gather is."""

  def __init__(self, recipe):
    if recipe.get('format') != 1:
      raise ValueError('a pre-embedded variant of another format (%s): make it again '
                       '(tools/onnx_preembed.py)' % recipe.get('format'))
    self.recipe = recipe
    self.parts = []
    for k, c in enumerate(recipe['concats']):
      p = dict(k=k, rank=c['rank'])
      if c.get('f'):
        f = c['f']
        p['f_src'] = [i for i, e in enumerate(f) if e['src'] is not None]
        p['f_const'] = [i for i, e in enumerate(f) if e['src'] is None]
        p['f_values'] = np.array([f[i]['value'] for i in p['f_const']], np.float32)
        p['f'] = f
      if c.get('v'):
        v = c['v']
        p['v_flag'] = [i for i, e in enumerate(v) if e.get('on') is not None]
        p['v_raw'] = [i for i, e in enumerate(v) if e.get('on') is None]
        p['v_on'] = np.array([v[i]['on'] for i in p['v_flag']], np.float32)
        p['v_off'] = np.array([v[i]['off'] for i in p['v_flag']], np.float32)
        p['v'] = v
      if c.get('oh'):
        rows = c['oh']['rows']
        p['oh'] = rows
        p['oh_depth'] = np.array([r['depth'] for r in rows], np.int64)
        p['oh_offset'] = np.array([r['offset'] for r in rows], np.int64)
        p['oh_zero'] = int(c['oh']['zero'])
      if c.get('emb'):
        e = c['emb']
        p['emb'] = e
        groups = collections.OrderedDict()
        for j, s in enumerate(e['sums']):
          key = json.dumps([s['prog']['ops'], s['prog']['consts'], s['prog']['outs']])
          groups.setdefault(key, []).append(j)
        p['groups'] = list(groups.values())
      self.parts.append(p)

  def __call__(self, feed):
    out = dict(feed)
    stacks = {}

    def src(names, rank):
      key = tuple(names)
      if key not in stacks:
        stacks[key] = (np.asarray(feed[names[0]]) if rank == 2 else
                       np.stack([np.asarray(feed[x]) for x in names], axis=1))
      return stacks[key]

    for p in self.parts:
      k, rank = p['k'], p['rank']
      lead = None
      if 'f' in p:
        cols = [src(p['f'][i]['src'], rank) for i in p['f_src']]
        lead = cols[0].shape if cols else lead
        if p['f_const']:
          lead = lead or self._lead(feed, p)
          x = np.empty(tuple(lead) + (len(p['f']),), np.float32)
          x[..., p['f_const']] = p['f_values']
          if cols:
            x[..., p['f_src']] = np.stack(cols, axis=-1)
        else:
          x = np.stack(cols, axis=-1).astype(np.float32, copy=False)
        out['pre%d_f' % k] = x
      if 'v' in p:
        lead = lead or self._lead(feed, p)
        x = np.empty(tuple(lead) + (len(p['v']),), np.float32)
        if p['v_flag']:
          flags = np.stack([src(p['v'][i]['src'], rank) for i in p['v_flag']], axis=-1)
          x[..., p['v_flag']] = np.where(flags, p['v_on'], p['v_off'])
        if p['v_raw']:
          x[..., p['v_raw']] = np.stack([src(p['v'][i]['src'], rank) for i in p['v_raw']], axis=-1)
        out['pre%d_v' % k] = x
      if 'oh' in p:
        idx = np.stack([src(r['src'], rank) for r in p['oh']], axis=-1).astype(np.int64)
        d = p['oh_depth']
        ok = (idx >= -d) & (idx < d)
        idx = np.where(idx < 0, idx + d, idx)
        out['pre%d_oh' % k] = np.where(ok, p['oh_offset'] + idx, p['oh_zero'])
      if 'emb' in p:
        e = p['emb']
        plain = [self._rows(src(r['src'], rank), r['offset'], r['rows']) for r in e['plain']]
        a = [self._rows(src(s['src'], rank), s['offset'], s['rows']) for s in e['sums']]
        b = [None] * len(e['sums'])
        for group in p['groups']:
          prog = e['sums'][group[0]]['prog']
          args = [np.stack([src(e['sums'][j]['prog']['inputs'][m], rank) for j in group], axis=-1)
                  for m in range(len(prog['inputs']))]
          idx, cond = _run_program(prog, args)
          idx = np.broadcast_to(np.asarray(idx, np.int64), np.shape(cond))
          cond = np.asarray(cond, bool)
          for g, j in enumerate(group):
            s = e['sums'][j]
            i = idx[..., g]
            i = np.where(i < 0, i + s['rows_b'], i)
            if ((i < 0) | (i >= s['rows_b']))[cond[..., g]].any():
              raise ValueError('an embedding index is outside its table')
            b[j] = np.where(cond[..., g], s['offset_b'] + i, e['zero'])
        out['pre%d_emb' % k] = np.stack(plain + a + b, axis=-1)
    for m in self.recipe['masks']:
      x = src(m['src'], 2 if len(m['src']) == 1 else 3)
      out[m['name']] = np.asarray(x, bool)[..., None]
    return out

  @staticmethod
  def _rows(x, offset, rows):
    i = np.asarray(x).astype(np.int64)
    i = np.where(i < 0, i + rows, i)
    if ((i < 0) | (i >= rows)).any():
      raise ValueError('an embedding index is outside its table')
    return offset + i

  def _lead(self, feed, p):
    """The batch shape of a part with no field of its own to read it from."""
    for key in ('f', 'v', 'oh'):
      for e in p.get(key, []):
        if e.get('src'):
          x = np.asarray(feed[e['src'][0]])
          return x.shape if p['rank'] == 2 else x.shape + (len(e['src']),)
    for e in self.recipe['concats']:
      for key in ('f', 'v', 'oh'):
        for r in e.get(key, {}) if key != 'oh' else e.get('oh', {}).get('rows', []):
          if r.get('src'):
            return np.asarray(feed[r['src'][0]]).shape
    raise ValueError('the variant reads no field to take the batch size from')


def _variant_matches(session, path, meta):
  """The variant was made from this very export (its size and export time,
  kept in the variant's metadata): a variant left over from an older export
  would read with old weights."""
  got = session.get_modelmeta().custom_metadata_map or {}
  return (got.get('from_exported') == str(meta.get('exported')) and
          got.get('from_bytes') == str(os.path.getsize(path)))


class Model:
  """One exported network and its .json.

  device: None or 'cpu' for the CPU; 'cuda' (or 'cuda:N') for an NVIDIA GPU,
  which needs onnxruntime-gpu - a separate environment from the app's own
  (docs: INSTALL.md, "s2 on the GPU"). A session that cannot put the network
  on the GPU is an error, not a quiet fall back to the CPU."""

  lstm = False                       # the LSTM-op variant (lstm_variant) is what runs
  pre = None                         # the pre-embedded variant's PreEmbed, when it runs

  def __init__(self, path, threads=None, optimization=None, device=None, mem_limit=None,
               spin=None, lstm=False, preembed=False):
    """lstm: on a card, run the export's LSTM-op variant when there is one
    (lstm_variant): a whole read in one call instead of a Loop step a frame.
    Only for reads that reset at their first frame (run()).
    preembed: on a card, run the step export's pre-embedded variant when there
    is one (preembed_variant): the same numbers from a quarter of the nodes;
    run() and prepare() turn the usual feed into its inputs."""
    import onnxruntime as ort
    self.path = path
    self.device = card_device(device or 'cpu')
    providers = _providers(self.device, mem_limit)
    self.meta = read_meta(path)
    variant = None
    if self.device != 'cpu':
      variant = lstm_variant(path) if lstm else preembed_variant(path) if preembed else None
    so = ort.SessionOptions()
    so.log_severity_level = 3        # it warns about every initializer it prunes
    if self.device.startswith('dml'):
      # DirectML runs a session's nodes in order and plans no memory patterns
      # (onnxruntime's DirectML provider notes)
      so.enable_mem_pattern = False
      so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    if threads:
      so.intra_op_num_threads = int(threads)
    if spin is not None:
      # spin=False: the thread pool's threads sleep between calls instead of
      # spinning for the next one. Spinning wins a little on an idle machine
      # and loses a lot on a busy one - two sessions of eight spinning
      # threads each, beside the simulator and the player's own programs
      so.add_session_config_entry('session.intra_op.allow_spinning', '1' if spin else '0')
    if optimization:
      # 'basic': Phillip's one-frame graphs run faster without the extended
      # rewrites (measured: 2.3 ms against 3.7 a frame, tools/bench_agent.py)
      so.graph_optimization_level = {
          'basic': ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
          'extended': ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
          'all': ort.GraphOptimizationLevel.ORT_ENABLE_ALL}[optimization]
    self.session = ort.InferenceSession(variant or path, so, providers=providers)
    if variant:
      if _variant_matches(self.session, path, self.meta):
        if lstm:
          self.lstm = True
        else:
          self.pre = PreEmbed(json.loads(
              self.session.get_modelmeta().custom_metadata_map['preembed']))
        path = variant
      else:                            # an old variant: the export itself
        self.session = ort.InferenceSession(path, so, providers=providers)
    if self.device.startswith('dml'):
      if 'DmlExecutionProvider' not in self.session.get_providers():
        raise RuntimeError('the DirectML provider did not load (%s): the card could not '
                           'take the model' % self.session.get_providers())
    elif self.device != 'cpu':
      got = self.session.get_providers()
      opts = self.session.get_provider_options().get('CUDAExecutionProvider', {})
      if 'CUDAExecutionProvider' not in got:
        raise RuntimeError('the CUDA provider did not load (%s): check the nvidia-* '
                           'wheels are in this environment' % got)
      if opts.get('use_tf32') != '0':
        raise RuntimeError('TF32 is on in the CUDA provider: the numbers would move')
    if self.device != 'cpu':
      self._check_card(path, so, providers)
    self.inputs = {i.name for i in self.session.get_inputs()}
    self.outputs = [o.name for o in self.session.get_outputs()]

  def _check_card(self, path, so, providers):
    """The card's answer to a small fixed input, against the processor's to
    the same input (self_check): a card that gives wrong numbers is never
    used. On DirectML a wrong answer is tried again without the card maker's
    own kernels (metacommands): an Intel UHD 770 gave the value function
    0.55 off with them and 1e-6 without (2026-10-01)."""
    import onnxruntime as ort
    same_game = dict(source=self.path, prepare=self.pre) if self.pre is not None else {}
    worst = self_check(path, self.session, **same_game)
    if worst <= CHECK_TOL:
      return
    if self.device.startswith('dml'):
      opts = dict(providers[0][1], disable_metacommands=True)
      session = ort.InferenceSession(path, so, providers=[('DmlExecutionProvider', opts)] +
                                     providers[1:])
      again = self_check(path, session, **same_game)
      if again <= CHECK_TOL:
        self.session = session
        self.no_metacommands = True
        return
      worst = again
    raise RuntimeError('the graphics card gives wrong numbers for %s (%.3g off the '
                       "processor's on a fixed input): it is not used"
                       % (os.path.basename(path), worst))

  @property
  def kind(self):
    return self.meta['kind']

  def observation_config(self):
    return observation_config(self.meta)

  def initial_state(self, batch):
    """The recurrent state a game starts from: zeros, as slippi-ai's LSTMs
    start (the exporter checks that it is)."""
    return {s['name']: np.zeros([batch] + s['shape'], s['dtype'])
            for s in self.meta['state']}

  def prepare(self, feed):
    """The feed as this session's inputs: the pre-embedded variant's stacked
    fields and table rows (PreEmbed), else as it is."""
    return feed if self.pre is None else self.pre(feed)

  def run(self, feed):
    if self.lstm:
      feed = self._from_start(feed)
    feed = self.prepare(feed)
    # an input the graph never reads is pruned from it; feeding it is an error
    got = self.session.run(None, {k: v for k, v in feed.items() if k in self.inputs})
    return dict(zip(self.outputs, got))

  def _from_start(self, feed):
    """The LSTM op has no reset: a row reset at the first frame starts from
    the zero state (what the export's reset puts there - tools/onnx_lstm.py
    checks), and a reset later in a row cannot be read this way."""
    reset = feed.get('reset')
    if reset is None:
      return feed
    reset = np.asarray(reset, bool)
    if reset[1:].any():
      raise ValueError('the LSTM-op variant reads rows from their start only')
    if not reset[0].any():
      return feed
    out = dict(feed)
    for s in self.meta['state']:
      if s['name'] in out:
        x = np.array(out[s['name']], copy=True)
        x[reset[0]] = 0
        out[s['name']] = x
    return out

  def can_chunk(self):
    """The export gives its final recurrent state (exports from before
    2026-09-26 do not), so a read can be split in time."""
    return all('state_out_%d' % i in self.outputs for i in range(len(self.meta['state'])))

  def run_chunked(self, feed, out, chunk=None, on_chunk=None):
    """run(feed)[out], `chunk` frames at a time, each window starting from
    the state the one before it ended in - the same recurrence, split (on
    the processor the very same numbers: measured 2026-10-02). The memory is
    a window's rather than the whole read's (the input projection alone is
    [frames, rows, 4 x width] floats). Every input but the recurrent state
    is time-major, so each is cut along its first axis. on_chunk(done, T)
    after each window: how far the read is."""
    names = [s['name'] for s in self.meta['state']]
    T = len(feed['reset'])
    if not chunk or chunk >= T:
      got = self.run(feed)[out]
      if on_chunk:
        on_chunk(T, T)
      return got
    if not self.can_chunk():
      raise ValueError('%s has no state outputs to carry between chunks - export it '
                       'again (tools/export_onnx.py)' % os.path.basename(self.path))
    state = {k: feed[k] for k in names}
    parts = []
    for t0 in range(0, T, int(chunk)):
      f = {k: v[t0:t0 + chunk] for k, v in feed.items() if k not in state}
      f.update(state)
      got = self.run(f)
      parts.append(got[out])
      state = {k: got['state_out_%d' % i] for i, k in enumerate(names)}
      if on_chunk:
        on_chunk(min(T, t0 + int(chunk)), T)
    return np.concatenate(parts, axis=0)


class ValueNet(Model):
  """The value function: per-frame values for a batch of rows [T, B]."""

  def values(self, state_action, reset, chunk=None, on_chunk=None):
    enc = encode(self.meta['encode'], state_action)
    feed = dict(enc, reset=reset, **self.initial_state(reset.shape[1]))
    return self.run_chunked(feed, 'value', chunk, on_chunk)


class RatingNet(Model):
  """The rating-conditioned policy: how unlikely each frame's real input was.

  slippi-ai's imitation_loss lines states up with the input `delay` frames
  later and predicts each one from the one before; that shifting happens
  here in numpy rather than in the graph (the exporter could not express it
  for a game of any length), so the graph sees the pairs already lined up."""

  def nll(self, state_action, reset, chunk=None):
    """-log p(real input) per frame, [T - delay - 1, B]: the same numbers as
    slippi-ai's policy.imitation_loss on the same rows. chunk: run_chunked()."""
    enc = encode(self.meta['encode'], state_action)
    d = int(self.meta['delay'])
    T = reset.shape[0]
    u = T - d
    feed = {}
    for name, x in enc.items():
      if name.startswith('.state'):
        x = x[:u]
      else:                           # the action and the rating are delayed
        x = x[d:]
      feed['x' + name] = x[:-1]
      if name.startswith('.action'):
        feed['next' + name[len('.action'):]] = x[1:]
    feed['reset'] = reset[:u][:-1]
    feed.update(self.initial_state(reset.shape[1]))
    return self.run_chunked(feed, 'nll', chunk)


def load(path, threads=None, device=None, mem_limit=None, lstm=False):
  with open(meta_path(path), encoding='utf-8') as f:
    kind = json.load(f).get('kind')
  cls = {'value': ValueNet, 'rating': RatingNet}.get(kind, Model)
  return cls(path, threads=threads, device=device, mem_limit=mem_limit, lstm=lstm)
