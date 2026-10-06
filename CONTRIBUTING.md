# Contributing

## Setting up

Python 3.12 or newer, and [uv](https://docs.astral.sh/uv/) only if you change a
dependency.

```sh
python3 -m venv .venv
.venv/bin/pip install --require-hashes --requirement requirements/dev.txt
.venv/bin/pip install -e . --no-deps
```

`dev.txt` brings the headless build of OpenCV. To work on the tools that open a window,
install `dev-gui.txt` into a separate environment instead. The two builds both provide
`cv2` and cannot share one.

Before pushing:

```sh
.venv/bin/ruff format --check .
.venv/bin/ruff check .
.venv/bin/mypy
.venv/bin/pytest
```

CI runs the same commands.

The tests in `tests/contract` talk to a real hub and are skipped without one.
`scripts/contract-test.sh` installs the hub from a checkout beside this one
(`../pihome-hub`, or `PIHOME_HUB_DIR`), starts it with mock relays and a throwaway key,
and runs them. CI does the same against a pinned hub commit, and weekly against the
hub's `main`.

## Commit messages

Subjects follow [Conventional Commits](https://www.conventionalcommits.org):

```
type(optional-scope)!: description
```

The body explains *why* the change is right rather than restating the diff.

| Part | Rule |
| --- | --- |
| `type` | one of `build` `chore` `ci` `docs` `feat` `fix` `perf` `refactor` `revert` `style` `test` |
| `scope` | optional, lower case: `camera` `detect` `track` `triggers` `lights` `hub` `config` `cli` `deploy` `docs` `tests` `ci` |
| `!` | append to the type or scope for a breaking change, and explain it in a `BREAKING CHANGE:` footer |
| `description` | lower case, imperative, no trailing full stop, whole subject within 72 characters |
| body | separated by one blank line, wrapped at 72, present for anything not self-evident |

Footers, where they apply: `Fixes: #123`, `Refs: #123`, `BREAKING CHANGE: ...`.

```
fix(hub): stop reporting after a refused key instead of retrying

The hub counts failed keys per address and blocks the address after ten.
Retrying a wrong key would lock out every other sensor behind the same
router, so one refusal now stops reporting and says so in the journal.

Fixes: #12
```

### Enable the hook

Once per clone:

```bash
git config core.hooksPath .githooks
git config commit.template .gitmessage
```

`.githooks/commit-msg` rejects a message that does not fit. It is a plain POSIX shell
script, and CI runs that same file over every commit in a pull request, so there is one
definition of valid, in one place.

## Dependencies

Direct versions are declared in `pyproject.toml`. Everything actually installed,
transitive packages included, is pinned with hashes in `requirements/`, one file per
install shape:

| File | Contents | Used by |
| --- | --- | --- |
| `headless.txt` | Runtime, headless OpenCV | The service |
| `gui.txt` | Runtime, OpenCV with windows | Drawing zones and watching the live picture |
| `dev.txt` | Runtime, headless OpenCV and the tooling | CI, and a development checkout |
| `dev-gui.txt` | Runtime, OpenCV with windows and the tooling | The CI leg that type-checks against the windowed build |

After changing a dependency in `pyproject.toml`, regenerate them:

```sh
requirements/refresh.sh
```

That keeps every version already pinned and moves only what the change forces. To move
the tree on purpose, pass the flag through:

```sh
requirements/refresh.sh --upgrade
```

CI runs the same script and fails if the result differs from what is committed.
