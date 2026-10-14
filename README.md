# phillip-onnx

ONNX exports of **Phillip**, the Super Smash Bros. Melee agents from Vlad
Firoiu's [slippi-ai](https://github.com/vladfi1/slippi-ai), and of two of
slippi-ai's analysis networks. They are shared here with Vlad Firoiu's
permission. All credit for the models goes to him and to slippi-ai.

The files are the assets of a release. The newest set is
[v2](../../releases/tag/v2); each file downloads from

    https://github.com/briandle00/phillip-onnx/releases/download/v2/<file>

| Release | What it is | Size |
|---|---|---|
| [v2](../../releases/tag/v2) | the agents and the rating-conditioned policy with their weights stored as float16 (they still compute in float32); the value function as in v1 | 16.6 GB |
| [v1](../../releases/tag/v1) | the first set: every network with float32 weights | 32.9 GB |

Both have the same file names and the same networks; v1 stays as it is.

## What is here

| Files | What they are |
|---|---|
| `vf_grouped_tx1x1024.onnx` + `.onnx.json` | slippi-ai's value function: how a position is going for each player, frame by frame |
| `grouped_d21_tx3x1536_noname.onnx` + `.onnx.json` | slippi-ai's rating-conditioned policy: how likely each frame's inputs were at each level of play |
| `<agent>.step.onnx` + `.json` | a Phillip agent, one frame at a time: what it presses next (for playing) |
| `<agent>.onnx` + `.json` | the same agent over a whole game at once (for reading a game) |
| `<agent>.step.pre.onnx`, `<name>.lstm.onnx` | the same networks rearranged for a graphics card ([NOTES.md](NOTES.md)) |
| `models.json` | every file with its size, SHA-256 and what it is, and every agent with the characters it plays, its opponent and its reaction delay |

There are 127 agents: the multi-character nets at each level (`bronze`,
`silver`, `gold`, `plat`, `diamond`, `master`, `gm`, `super-gm`), and the
per-character ones - `<character>_d<delay>_vs_<opponent>` for one matchup,
`<character>_d<delay>_ditto` for the mirror, and
`<character>_d<delay>_imitation_v<n>`, trained on people only. Where a run
has several saves, only the newest is here.

Each `.json` beside a network says what it was exported from, its inputs and
outputs, its recurrent state, and how a slippi-ai game state is encoded into
its inputs. They run on [onnxruntime](https://onnxruntime.ai) with no
TensorFlow or JAX; the exports were checked against the original networks
on real games (the value function to within a few millionths, the agents'
outputs to within about 1e-3).

## float16-stored weights (v2)

In v2 every weight of 256 values or more in the agents and the
rating-conditioned policy is stored as float16, with one `Cast` back to
float32 that onnxruntime folds when the file loads: the inputs, the outputs
and every operation are float32 as before, on weights rounded to float16
(the largest change to any weight is 0.05% of it). Nothing is needed to load
them that v1 did not need. The files are half the size.

The networks are as good and not bit-for-bit the same: on a fixed input an
agent's outputs move by at most 0.92% of their size, and with the same
random draws about one sampled press in 3,000 differs from v1's. The
value function is float32 in v2 too: float16 moved its outputs in the fourth
decimal for a saving of 0.09 GB. The `.step.pre.onnx` files give the same
bits as their `.step.onnx`, as in v1.

`tools/onnx_f16.py` is the converter, with the checks each file passed.

## Checking a download

`models.json` lists every file's SHA-256. On Windows:

    certutil -hashfile gm.step.onnx SHA256

## Running them on a graphics card

[NOTES.md](NOTES.md) has what was learned running these on Windows: which
variant to load (`.step.pre.onnx` and `.lstm.onnx` are the same networks
rearranged for a card, several times faster there), the onnxruntime settings
that matter (TF32 off), and pinned package lists that need no system CUDA
(`requirements-cuda.txt`, `requirements-directml.txt`).

`tools/` has the scripts that make the two card variants from a plain export,
the small loader they share, and the float16 converter.

## Terms

The weights are Vlad Firoiu's, from slippi-ai, and are shared here with his
permission. slippi-ai's code is MIT-licensed. The scripts in `tools/` are
MIT-licensed too. If you use any of it, credit slippi-ai.
