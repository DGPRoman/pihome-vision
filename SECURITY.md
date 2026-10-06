# Security

This service watches a camera pointed at a house. Two things here are worth stealing:
the camera's own credentials, which let anybody watch it, and the hub's sensor key,
which lets anybody report motion and so switch whatever light a rule ties to it. The
notes below describe how the project keeps both, and the pictures, where they belong.

## Reporting a vulnerability

Please report privately rather than opening a public issue: use
[GitHub's private vulnerability reporting](https://github.com/DGPRoman/pihome-vision/security/advisories/new)
on this repository. A response should be expected within a week. This is a personal
project with no commercial support and no bug bounty.

## What it sees, and what leaves the machine

The project is at an early stage and not yet usable. What follows is how it is built,
and each part is held to it as it lands.

| Data | Where it goes |
| --- | --- |
| Camera frames | Held in memory, each replaced by the next. Never written to disk, never sent anywhere. The one exception is a command that saves a single frame to a file the person running it names, for drawing zones on |
| What was detected | Used to decide each trigger, then discarded. No history is kept |
| A trigger's state | `true` or `false`, sent to the hub as a sensor reading. That is all the hub learns |
| The camera's address and password | Read from the environment. Removed from any URL before it reaches a log line or an error message |
| The hub's sensor key | Read from the environment, sent only to the hub, and never logged |

There is no telemetry, and the model is a local file. Nothing is downloaded while the
service runs.

## What a stolen key can do

The hub's sensor key may only push readings. Its holder can report motion for any sensor
the hub knows, and so light whatever a rule ties to one. It cannot read anything, switch
a relay directly or administer the hub. A key that leaks is rotated in the hub's
configuration and in this service's environment file together.

## Handling secrets in this repository

- Never commit an environment file. `.env` and `*.env` are git-ignored, and
  `.env.example` uses placeholder values and documentation addresses.
- Keep camera addresses, zones and anything else that describes a real house in
  configuration, not in source. `config/*.yaml` is git-ignored; only `*.example.yaml`
  templates are tracked.
- Never commit model weights or frames. `*.onnx`, `models/` and image files are
  git-ignored. CI fails if an environment file, a model or a frame is ever tracked.
- If a secret is ever committed, treat it as public permanently and rotate it.
  Rewriting history does not retract what has already been fetched, forked or indexed.
