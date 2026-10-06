# pihome-vision

Watches a camera for people and vehicles. When somebody is in a chosen zone or crosses a
chosen line, it switches a light on through
[pihome-hub](https://github.com/DGPRoman/pihome-hub), and switches it off again a while
after they have gone.

**Status:** being built, not yet usable. The work is tracked in this repository's issues
and on the [Smart Home](https://github.com/users/DGPRoman/projects/3) board.

## How it fits with the hub

Everything is decided here: which light, for how long, and whether only after dark. The
hub is told only to switch a relay on, and later off, with its relay key. A light that
somebody switched on by hand is left alone.

## Configuration

Secrets come from the environment, and everything else from `vision.yaml`:

| Where | What | Template |
| --- | --- | --- |
| Environment | The camera's address with its password, the hub's address and its relay key | [`.env.example`](.env.example) |
| `vision.yaml` | The model, the camera's zones and lines, which lights follow them, and where the house is for sunset | [`config/vision.example.yaml`](config/vision.example.yaml) |

`pihome-vision validate` reads both and says what they describe, with the camera's
password left out. A problem in either exits with status 2, naming each variable or
field that is wrong.

## The lights

A light is on while any of its zones is occupied, and for `off_after_seconds` after the
last of them comes clear or one of its lines is crossed. With `only_after_dark` it comes
on only between sunset and sunrise at the configured location.

Before switching a light on, the service reads it from the hub. One that is on already
was switched on by somebody else, and is left alone. Before switching it off, the
service reads it again, so a light somebody switched off in the meantime is not touched.
A light the hub cannot be reached to switch on is given up after ten seconds rather than
lit for an empty yard later.

Switching a relay through the hub's API cancels any countdown one of the hub's own rules
had running on it, so give a light to the hub's rules or to this service, not both.

## The camera

ffmpeg reads the camera, so it has to be installed (`apt install ffmpeg` on Debian or
Ubuntu). An `rtsp://`, `rtsps://`, `http://` or `https://` address works, and so does
`cam:0` for the first local webcam. A stream that drops or goes quiet is reopened, with
a wait that grows to 30 seconds between attempts.

`pihome-vision check` connects once, says how big the frames are and how fast they
arrive, and runs the model on one of them. When it cannot connect it names the likely
reason, such as a refused connection, no answer or a wrong password, and exits with
status 1.

## First setup

Zones and lines are drawn on a desktop, with the build of OpenCV that opens windows.
The service itself needs none, and on an always-on machine takes
`requirements/headless.txt` instead.

```console
$ python3 -m venv .venv
$ .venv/bin/pip install --require-hashes --requirement requirements/gui.txt
$ .venv/bin/pip install --no-deps .
$ cp .env.example .env
$ cp config/vision.example.yaml config/vision.yaml
```

Put the camera's address, the hub's and its relay key in `.env`, and a model in
`config/vision.yaml` (see [the model](#the-model)). Then, from the same directory:

1. `pihome-vision check` connects to the camera and runs the model on one frame.
2. `pihome-vision snapshot gate.jpg` saves a frame, readable only by you, to draw on
   later or somewhere else. `edit` takes its own frame when not given one.
3. `pihome-vision edit`, or `edit --image gate.jpg`, opens the frame with the zones
   and lines already in `vision.yaml`. Draw on the ground, where people stand:

   | Do | To |
   | --- | --- |
   | Click | Add a point |
   | `z` | Close the points into a zone |
   | `l` | Make the two points a line |
   | Type, then Enter | Name it. Enter alone keeps the name offered |
   | Backspace | Take back the last point, or with none, the last shape |
   | Esc | Go back from naming to the points |
   | `q` | Finish |

   It prints the `cameras:` section with what was drawn. Paste it over the one in
   `vision.yaml`, then set each trigger's classes and times, and each line's direction,
   there. A light that names a trigger no longer drawn is pointed out.
4. `pihome-vision validate` checks the result.
5. `pihome-vision preview` shows the live picture with what the model finds, where
   each object stands, which zones are occupied and which lines were just crossed.
   Nothing is switched. `q` closes it.
6. `pihome-vision run` starts switching the lights.

## Running

`pihome-vision run` watches the camera and switches the lights until it is stopped. It
logs each trigger as it changes and each light as it is switched, and once a minute
says how many frames arrived, how many the model looked at and how long that took. With
`motion_threshold` above 0, a picture that has hardly changed is not shown to the model
again, for up to five seconds.

SIGTERM or Ctrl-C switches off every light it switched on, and then it exits. A camera
that stops sending frames lets its zones come clear and its lights go off in the usual
time. A configuration it cannot run with, including a model that will not load, exits
with status 2. Any other failure exits with status 1.

## The model

No model comes with this repository, and each has its own licence.
[docs/models.md](docs/models.md) says which ones work, where to get one, and how to try it
on a photo before pointing it at the camera:

```console
$ pihome-vision detect photo.jpg
```

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md). [SECURITY.md](SECURITY.md) says what the service
sees and what leaves the machine.

## License

[MIT](LICENSE)
