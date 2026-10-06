#!/usr/bin/env bash
#
# Provision pihome-vision as a systemd service, or upgrade an install already in place.
#
# Paths are read out of pihome-vision.service, so this script cannot drift away from the
# unit it installs. Re-running is safe: an edited environment file or vision.yaml is
# never overwritten.
#
# Usage: sudo deploy/install.sh

set -euo pipefail

DEPLOY_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
UNIT_SOURCE="$DEPLOY_DIR/pihome-vision.service"
UNIT_NAME="$(basename -- "$UNIT_SOURCE")"
REPO_ROOT="$(dirname -- "$DEPLOY_DIR")"
readonly DEPLOY_DIR UNIT_SOURCE UNIT_NAME REPO_ROOT

# The service's lockfile: OpenCV without windows, which a service has no use for.
readonly LOCKFILE=requirements/headless.txt

say() { printf '==> %s\n' "$*"; }
die() {
    printf 'install.sh: %s\n' "$*" >&2
    exit 1
}

[[ -f $UNIT_SOURCE ]] || die "$UNIT_SOURCE is missing"

# The last assignment wins, the way systemd itself reads a unit file.
unit_value() {
    local value
    value="$(sed -n "s/^$1=//p" "$UNIT_SOURCE" | tail -n 1)"
    [[ -n $value ]] || die "$UNIT_NAME declares no $1= — cannot tell where to install"
    printf '%s\n' "$value"
}

ENV_FILE="$(unit_value EnvironmentFile)"
INSTALL_ROOT="$(unit_value WorkingDirectory)"
# ExecStart= is the program and its arguments; the venv is two levels above the program.
VENV="$(dirname -- "$(dirname -- "$(unit_value ExecStart | cut -d ' ' -f 1)")")"
# LoadCredential=NAME:PATH, and the file is PATH.
CONFIG_FILE="$(unit_value LoadCredential | cut -d : -f 2-)"
CONFIG_DIR="$(dirname -- "$ENV_FILE")"
readonly ENV_FILE INSTALL_ROOT VENV CONFIG_FILE CONFIG_DIR

# -- Preflight ---------------------------------------------------------------

[[ $EUID -eq 0 ]] || die "run as root: sudo $0"
command -v systemctl >/dev/null || die "no systemctl — this installs a systemd service"
python3 -c 'import sys; sys.exit(sys.version_info < (3, 12))' ||
    die "pihome-vision needs Python 3.12 or newer, and python3 here is $(python3 -V 2>&1)"
python3 -c 'import venv, ensurepip' >/dev/null 2>&1 ||
    die "the python3 venv module is missing (apt install python3-venv)"
# The service finds ffmpeg on systemd's PATH, which is not necessarily root's.
[[ -x /usr/bin/ffmpeg || -x /usr/local/bin/ffmpeg ]] ||
    die "ffmpeg is not installed where the service will look for it (apt install ffmpeg)"

[[ $REPO_ROOT == "$INSTALL_ROOT" ]] || die "$UNIT_NAME runs the service from $INSTALL_ROOT,
but this checkout is at $REPO_ROOT. Clone it there, or point WorkingDirectory=
and ExecStart= at this path instead."

say "installing from $REPO_ROOT"

# -- Package -----------------------------------------------------------------

if [[ ! -x $VENV/bin/pip ]]; then
    say "creating $VENV"
    python3 -m venv "$VENV"
fi

# Dependencies come from the lockfile, with hashes, so a machine set up months after
# CI last ran gets the set CI tested rather than whatever resolves that day.
# --no-deps on the package itself: everything it needs is already pinned above, and
# without it pip is free to re-resolve and undo the lock.
say "installing dependencies from $LOCKFILE"
"$VENV/bin/pip" install --quiet --require-hashes --requirement "$REPO_ROOT/$LOCKFILE"

# Installed, not linked with -e, so pip byte-compiles once here rather than the
# service recompiling on every start against a read-only filesystem.
say "installing the package into $VENV"
"$VENV/bin/pip" install --quiet --upgrade --no-deps "$REPO_ROOT"

# -- Configuration -----------------------------------------------------------

# root's alone: systemd reads both files as PID 1 and hands the service what is in
# them, so no other account, the service's included, needs to read them here.
install -d -m 700 -o root -g root "$CONFIG_DIR"

fresh=()

if [[ -e $ENV_FILE ]]; then
    say "keeping the existing $ENV_FILE"
else
    say "writing $ENV_FILE from .env.example"
    # The unit says where vision.yaml is, so the example's line about it is left out.
    install -m 600 -o root -g root /dev/null "$ENV_FILE"
    grep -v '^# *PIHOME_VISION_CONFIG_PATH=' "$REPO_ROOT/.env.example" >"$ENV_FILE"
    fresh+=("$ENV_FILE")
fi

if [[ -e $CONFIG_FILE ]]; then
    say "keeping the existing $CONFIG_FILE"
else
    say "copying the example to $CONFIG_FILE"
    install -m 600 -o root -g root "$REPO_ROOT/config/vision.example.yaml" "$CONFIG_FILE"
    fresh+=("$CONFIG_FILE")
fi

install -d -m 755 -o root -g root "$INSTALL_ROOT/models"

# -- Unit --------------------------------------------------------------------

say "installing $UNIT_NAME"
install -m 644 -o root -g root "$UNIT_SOURCE" "/etc/systemd/system/$UNIT_NAME"
systemctl daemon-reload
systemctl enable "$UNIT_NAME" >/dev/null

# The examples name a camera, a hub and a key that are not yours, and zones drawn over
# somebody else's driveway. A fresh install stops here and says what to fill in.
if ((${#fresh[@]})); then
    printf '\nEnabled but not started: these are still the examples.\n\n'
    for file in "${fresh[@]}"; do
        printf '  sudoedit %s\n' "$file"
    done
    cat <<EOF

Put the camera's address and the hub's relay key in $ENV_FILE, the zones
from pihome-vision edit in $CONFIG_FILE, and the model at the path it
names, relative to $INSTALL_ROOT. Then start it:

  systemctl start $UNIT_NAME
  journalctl -u $UNIT_NAME -f
EOF
    exit 0
fi

say "restarting $UNIT_NAME"
# Not fatal here: a unit that fails to start is what the check below explains.
systemctl restart "$UNIT_NAME" || true

# -- Verify ------------------------------------------------------------------

# Up is not enough: a configuration or a model it cannot use is an exit a few seconds
# after starting. A camera it cannot reach is not: that is retried, and logged.
say "watching it for 10 seconds"
sleep 10
if systemctl is-active --quiet "$UNIT_NAME" &&
    [[ $(systemctl show --property=NRestarts --value "$UNIT_NAME") == 0 ]]; then
    say "pihome-vision is running: journalctl -u $UNIT_NAME -f"
    exit 0
fi

printf 'install.sh: %s did not stay up\n' "$UNIT_NAME" >&2
systemctl --no-pager --full status "$UNIT_NAME" || true
journalctl -u "$UNIT_NAME" -n 30 --no-pager || true
exit 1
