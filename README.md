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
