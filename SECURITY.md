# Security

This service watches a camera pointed at a house and switches the house's lights. Two
things here are worth stealing: the camera's own credentials, which let anybody watch
it, and the hub's relay key, which lets anybody switch every relay the hub has. The
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
| Camera frames | Held in memory, each replaced by the next. Never sent anywhere, and never written to disk except by `snapshot`, which saves one frame to a file the person running it names, readable only by them. `edit` and `preview` show frames in a window on the machine's own screen |
| What was detected | Used to decide each trigger and each light, then discarded. No history is kept |
| A light's state | Switch this relay on, or off. That is all the hub learns |
| The camera's address and password | Read from the environment and handed to ffmpeg, which reads the camera. Removed from any URL, and from anything ffmpeg says, before it reaches a log line or an error message. ffmpeg is given the address on its command line, which every account on the machine can read from `/proc`: run this where nobody else has an account, or mount `/proc` with `hidepid=invisible` |
| The hub's relay key | Read from the environment, sent only to the hub's address, and never logged. A redirect is not followed, and a proxy set in the environment is not used, since either would carry the key somewhere else. Plain HTTP is refused unless the hub's address is a private one |

There is no telemetry, and the model is a local file. Nothing is downloaded while the
service runs.

## What a stolen key can do

The hub has no key narrower than the one this needs. Its relay key reads and switches
every relay, so whoever holds it can switch any light, or anything else on a relay, in
the house. It cannot administer the hub's accounts. Keep it in one file, readable only
by the account the service runs as. If it leaks, generate a new one in the hub's
configuration and put it here, and every other client that uses it, at the same time.

A key the hub turns down is not tried again until the service restarts. The hub counts
failed keys per address, and a service retrying a stale key would soon have it refuse
every client behind the same router.

## What the lights do when it is not running

This service decides when a light goes off, so a light it switched on stays on until it
says otherwise. Stopping the service, or shutting the machine down normally, switches
off every light it switched on. A crash or a power cut cannot, and a light lit at that
moment stays lit until somebody switches it off. A light that somebody else switched on
is left alone, then and at any other time.

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
