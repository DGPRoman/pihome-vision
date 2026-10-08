# pihome-vision

Watches a camera for people and vehicles. When somebody is in a chosen zone or crosses a
chosen line, it switches a light on through
[pihome-hub](https://github.com/DGPRoman/pihome-hub), and switches it off again a while
after they have gone.

**Status:** usable with one camera, run by hand or as a systemd service. The work is
tracked in this repository's issues and on the
[Smart Home](https://github.com/users/DGPRoman/projects/3) board.

## How it fits with the hub

Everything is decided here: which light, for how long, and whether only after dark. The
hub is told only to switch a relay on, and later off, with its relay key. A light that
somebody switched on by hand is left alone, and so is one whose automation somebody has
turned off in the hub.

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

The same reads say whether somebody has turned the light's automation off in the hub.
While it is off the light is theirs: the service neither switches it on nor off, and
says so in its log once each time it would have. A hub too old to say is taken as
leaving automation on.

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
The service itself needs none, and is deployed with the headless build (see
[deploying](#deploying)).

```console
$ python3 -m venv .venv
$ .venv/bin/pip install --require-hashes --requirement requirements/gui.txt
$ .venv/bin/pip install --no-deps .
$ cp .env.example .env
$ cp config/vision.example.yaml config/vision.yaml
```

Put the camera's address in `.env`, and a model in `config/vision.yaml` (see
[the model](#the-model)). The hub's address and its relay key are needed only by
`validate` and `run`, so a desktop used only to draw zones need not hold the key to
every relay. Then, from the same directory:

1. `pihome-vision check` connects to the camera and runs the model on one frame.
2. `pihome-vision snapshot gate.jpg` saves a frame, readable only by you, to draw on
   later or somewhere else. `edit` takes its own frame when not given one.
3. `pihome-vision edit`, or `edit --image gate.jpg`, opens the frame with the zones
   and lines already in `vision.yaml`. Any part of somebody counts, a hand over a
   zone's edge or a shoulder on a line, so draw them where people will show up in the
   picture:

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

A camera with `only_after_dark` is watched only between sunset and sunrise at the
configured location. By day its stream is closed and the model does not run, which on
an always-on PC saves about two CPU cores, so set it whenever every light the camera
switches is `only_after_dark` as well. The service looks at the sun once a minute and
logs when watching stops and starts again. At sunrise a zone still occupied comes clear
and its light goes off in the usual time, as for a camera that stops sending frames.

SIGTERM or Ctrl-C switches off every light it switched on, and then it exits. A camera
that stops sending frames lets its zones come clear and its lights go off in the usual
time. A configuration it cannot run with, including a model that will not load, exits
with status 2. Any other failure exits with status 1. Deployed as below, a pipeline that
stops moving for 15 seconds, stuck in a model run say, is stopped the same way, with
its lights switched off, and started again.

## Tuning

How soon a light comes on, and how often it misses somebody, depends mostly on these.

**The camera's stream.** Most cameras send a main stream at full resolution and a
lighter sub stream. Give the service the main one. The model sees a 640-wide picture
either way, but one scaled down from the full frame is sharper and less noisy than the
sub stream's own, at night above all: on one camera after dark, the person in view was
found in 80-91% of the frames from the main stream and in 47-52% of those from the sub
stream. Decoding the main stream costs more, about a third of a core for 2560x1440 at
ten frames a second on a desktop CPU from 2012.

**`fps`**, per camera. The frames the model looks at each second. Somebody who comes
into view waits half the gap between two of them on average before the model sees them:
100 ms at 5, 50 ms at 10. Each frame costs the model the same, so 10 take twice the CPU
of 5. Only the newest frame is kept, so a model that cannot keep up skips the rest: the
line logged once a minute then shows fewer frames examined than came from the camera.

**`confidence`**, under `model`. Detections less certain than this are ignored. Lower
finds somebody in more frames, in the dark especially, and lets more shadows and bushes
through as people. `pihome-vision detect` prints the confidence of everything it finds,
so try it on a few frames saved with `snapshot` by day and by night before changing it.

**`motion_threshold`**, per camera. Above 0, a picture that has hardly changed is not
shown to the model again for up to five seconds, which saves CPU while nothing moves.
The change is averaged over the whole frame, so set too high it lets somebody small and
slow at the edge of the picture wait out those five seconds.

**`threads` and `engine`**, under `model`. Which engine runs the model, and on how many
threads: [docs/models.md](docs/models.md#engine) compares them. So does the model's
size, in [choosing a size](docs/models.md#choosing-a-size).

## Deploying

`deploy/install.sh` installs the service on a machine that stays on, as a systemd unit
that starts with it. It needs Python 3.12 or newer, `python3-venv` and ffmpeg (on
Debian or Ubuntu, `apt install python3-venv ffmpeg`), and the checkout at
`/opt/pihome-vision`:

```console
$ sudo git clone https://github.com/DGPRoman/pihome-vision /opt/pihome-vision
$ sudo /opt/pihome-vision/deploy/install.sh
```

It installs from the hashed lockfile into `/opt/pihome-vision/.venv`, and the first
time copies the examples to two files only root can read:

| File | What |
| --- | --- |
| `/etc/pihome-vision/vision.env` | The camera's address with its password, the hub's address and its relay key |
| `/etc/pihome-vision/vision.yaml` | The model, zones, lines, lights and location, as drawn with `edit` |

The service is then enabled but not started, since neither file is yours yet. The model
goes where `vision.yaml` says, relative to `/opt/pihome-vision`: the example's
`models/detector.onnx` is the checkout's `models/` directory. Fill in both files, then:

```console
$ sudo systemctl start pihome-vision
$ journalctl -u pihome-vision -f
```

The relay key is `PIHOME_RELAY_API_KEY` from the hub's own environment, in
`/etc/pihome-hub/hub.env` on a hub installed by its installer. It switches every relay
the hub has, so keep it in `vision.env` and nowhere else. [SECURITY.md](SECURITY.md)
says what it can do and what to do if it leaks.

To upgrade, `git pull` in the checkout and run the installer again. It keeps both
files, restarts the service and watches it for ten seconds. A configuration or model it
cannot use stops the service for good, with the reason in the journal: restarting
would not change it.

The unit runs as an account made up for each run. It can write nowhere, sees no
devices, home directories or other processes, and holds no privileges; the network is
the one thing it can reach. `vision.yaml` is handed to it by systemd, so it can stay
root's. A local webcam (`cam:0`) needs the unit changed
to let the device through; its comments say how.

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
