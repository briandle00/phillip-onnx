# Running these on a graphics card: notes

What was learned getting the exports to run fast on Windows with
[onnxruntime](https://onnxruntime.ai). Numbers are from one PC (RTX 4070),
measured 2026-10-02; treat them as a guide, not a benchmark.

## Which file to load

| File | Use it for | Why |
|---|---|---|
| `<agent>.step.pre.onnx` | playing a frame at a time on a card | the input encoding (618 of `gm`'s 821 nodes) is folded into about 30 nodes; a step of 64 rows went from 7.2 ms to roughly its arithmetic, 1.2 ms |
| `<agent>.step.onnx` | playing a frame at a time on a processor, or anything that wants one input a game field | the plain export |
| `<name>.lstm.onnx` | reading a whole game on a card | each LSTM layer is ONNX's LSTM op (one cuDNN call) instead of a Loop; one game's value read went from 2.9 s to 0.14 s |
| `<name>.onnx` | reading a whole game on a processor | the Loop form: 5.1 s a game on four threads |

On a card every node is a kernel launch, and a Loop's control runs on the
host. That is the whole reason for the two variants: same weights, same
outputs, far fewer launches.

The `.pre` variant takes a handful of stacked inputs instead of one input a
field. The recipe for building them from the usual feed is in the file's own
metadata (key `preembed`); `tools/onnx_models.py` has a small class,
`PreEmbed`, that follows it. The `.lstm` variant has no `reset` input: zero a
row's state yourself on its first frame.

## Settings that matter

- **Turn TF32 off on NVIDIA.** onnxruntime's CUDA provider has it on by
  default (`use_tf32` reads 1), which rounds matrix multiplies and moves the
  outputs. Pass `{'use_tf32': '0'}` in the provider options.
- **`arena_extend_strategy: kSameAsRequested`** keeps the CUDA memory arena
  from doubling.
- **One session per card, batch the rows.** Several sessions on one card
  fight each other; one session with a bigger batch is faster.
- **DirectML** (`onnxruntime-directml`) runs on any DirectX 12 card. It works
  on Intel integrated graphics but is slow there. AMD is untested: no
  hardware here. Reports are welcome.

## Package versions that work together

`requirements-cuda.txt` and `requirements-directml.txt` are pinned, with
hashes, and install into a plain virtual environment with no system CUDA:

    python -m venv gpu-env
    gpu-env\Scripts\pip install --require-hashes -r requirements-cuda.txt

The NVIDIA set is about 1.3 GB (onnxruntime-gpu plus NVIDIA's own CUDA 13 and
cuDNN 9 wheels from PyPI). onnxruntime finds the libraries inside the
environment if their `bin` folders are on the DLL search path before the
first session is made (`os.add_dll_directory` on each `nvidia\*\bin`).
Only the graphics driver has to be on the PC already.

## The scripts in `tools/`

- `onnx_preembed.py PATH\<agent>.step.onnx` writes `<agent>.step.pre.onnx`.
  Checked before writing: identical outputs (difference exactly 0) on seeded
  inputs over each field's whole range.
- `onnx_lstm.py PATH\<name>.onnx` writes `<name>.lstm.onnx`. Checked before
  writing: outputs within 1e-4 on the processor.
- `onnx_models.py` is the small loader both use: session options, the
  provider choice, `PreEmbed`, state handling.

They need `numpy`, `onnx` and `onnxruntime`. They work on the exports in this
repository's release; the exporter that makes those from slippi-ai's
checkpoints is not here yet.
