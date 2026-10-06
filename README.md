# pihome-vision

Watches a camera for people and vehicles and tells
[pihome-hub](https://github.com/DGPRoman/pihome-hub) when somebody is in a chosen zone or
crosses a chosen line, so the hub's rules can switch the lights.

**Status:** being built, not yet usable. The work is tracked in this repository's issues
and on the [Smart Home](https://github.com/users/DGPRoman/projects/3) board.

## How it fits with the hub

Each zone or line is reported to the hub as one of its sensors: `{"motion": true}` when
it starts, `{"motion": false}` when it stops. The hub decides what that means, which
light, for how long and whether only after dark, with the rules it already has. This
service only says what the camera sees.

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md). [SECURITY.md](SECURITY.md) says what the service
sees and what leaves the machine.

## License

[MIT](LICENSE)
