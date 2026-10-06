#!/usr/bin/env bash
# Run the contract tests against a real hub, started here and stopped when they
# finish.
#
#   scripts/contract-test.sh [extra pytest arguments]
#
# The hub comes from a checkout of pihome-hub: PIHOME_HUB_DIR, or ../pihome-hub
# beside this repository. It is installed into a virtualenv under build/ from its
# hashed lockfile, so the run proves the versions the hub ships with. HUB_PYTHON
# picks the hub's interpreter, python3.11 by default when there is one, as the hub's
# own CI; PYTHON picks this project's, .venv/bin/python by default when there is one.
#
# Each run gets a hub of its own: a new relay key, a new database, the mock relay
# backend, loopback only, a port nobody else has. Nothing is read from or written to
# the checkout's .env or the developer's database.
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
hub_dir="$(cd "${PIHOME_HUB_DIR:-$root/../pihome-hub}" && pwd)"
hub_python="${HUB_PYTHON:-$(command -v python3.11 || command -v python3)}"
if [ -z "${PYTHON:-}" ] && [ -x "$root/.venv/bin/python" ]; then
    PYTHON="$root/.venv/bin/python"
fi
python="${PYTHON:-python3}"
venv="$root/build/contract/venv"

work="$(mktemp -d)"
hub_pid=""

finish() {
    status=$?
    if [ -n "$hub_pid" ]; then
        kill "$hub_pid" 2>/dev/null || true
        wait "$hub_pid" 2>/dev/null || true
    fi
    if [ "$status" -ne 0 ] && [ -f "$work/hub.log" ]; then
        echo "--- the hub's log ---" >&2
        tail -n 100 "$work/hub.log" >&2
    fi
    rm -rf "$work"
}
trap finish EXIT

echo "installing the hub from $hub_dir"
if [ ! -x "$venv/bin/python" ]; then
    "$hub_python" -m venv "$venv"
fi
"$venv/bin/pip" install --quiet --disable-pip-version-check --require-hashes \
    --requirement "$hub_dir/requirements/base.txt"
"$venv/bin/pip" install --quiet --disable-pip-version-check --no-deps --editable "$hub_dir"

secret() {
    "$venv/bin/python" -c 'import secrets; print(secrets.token_urlsafe(48))'
}

port="$("$venv/bin/python" -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])')"
relay_key="$(secret)"

# From the work directory, so that no .env is found and read, and the sensor,
# automation and device files the hub looks for there do not exist: only relays are
# under test. STATE_DIRECTORY is what a systemd unit would set, and is cleared.
(
    cd "$work"
    unset STATE_DIRECTORY
    exec env \
        PIHOME_HOST=127.0.0.1 \
        PIHOME_PORT="$port" \
        PIHOME_RELAY_API_KEY="$relay_key" \
        PIHOME_SENSOR_API_KEY="$(secret)" \
        PIHOME_GPIO_BACKEND=mock \
        PIHOME_RELAY_CONFIG_PATH="$root/tests/contract/relays.yaml" \
        PIHOME_DATABASE_PATH="$work/hub.db" \
        PIHOME_ACCESS_LOG=true \
        "$venv/bin/pihome-hub"
) >"$work/hub.log" 2>&1 &
hub_pid=$!

origin="http://127.0.0.1:$port"
echo "waiting for the hub at $origin"
for _ in $(seq 60); do
    if curl --silent --fail --output /dev/null "$origin/health"; then
        break
    fi
    if ! kill -0 "$hub_pid" 2>/dev/null; then
        echo "the hub stopped before it was ready" >&2
        exit 1
    fi
    sleep 0.5
done
curl --silent --fail --output /dev/null "$origin/health"

PIHOME_CONTRACT_HUB="$origin" \
    PIHOME_CONTRACT_RELAY_KEY="$relay_key" \
    "$python" -m pytest "$root/tests/contract" -p no:cacheprovider "$@"
