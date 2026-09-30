# phillip-onnx

ONNX exports of **Phillip**, the Super Smash Bros. Melee agents from Vlad
Firoiu's [slippi-ai](https://github.com/vladfi1/slippi-ai), and of two of
slippi-ai's analysis networks. They are shared here with Vlad Firoiu's
permission. All credit for the models goes to him and to slippi-ai.

The files are the assets of the [v1 release](../../releases/tag/v1). Each one
downloads from

    https://github.com/briandle00/phillip-onnx/releases/download/v1/<file>

## What is here

| Files | What they are |
|---|---|
| `vf_grouped_tx1x1024.onnx` + `.onnx.json` | slippi-ai's value function: how a position is going for each player, frame by frame |
| `grouped_d21_tx3x1536_noname.onnx` + `.onnx.json` | slippi-ai's rating-conditioned policy: how likely each frame's inputs were at each level of play |
| `<agent>.step.onnx` + `.json` | a Phillip agent, one frame at a time: what it presses next (for playing) |
| `<agent>.onnx` + `.json` | the same agent over a whole game at once (for reading a game) |
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

## Checking a download

`models.json` lists every file's SHA-256. On Windows:

    certutil -hashfile gm.step.onnx SHA256

## Terms

The weights are Vlad Firoiu's, from slippi-ai, and are shared here with his
permission. slippi-ai's code is MIT-licensed; this repository adds no code.
If you use them, credit slippi-ai.
