#!/usr/bin/env bash
# Regenerate the lockfiles from pyproject.toml.
#
# Without --upgrade, uv keeps the versions already pinned here as preferences and
# only moves what pyproject.toml forces it to, so a dependency change does not drag
# the whole tree forward with it. Pass --upgrade through to do that deliberately:
#
#     requirements/refresh.sh --upgrade
#
# The resolution is universal — one file per install shape, valid for every Python and
# platform this project supports. --python-version names the floor from
# requires-python, so the "# via" annotations do not depend on the interpreter that
# happens to run this.
#
# Every shape takes exactly one of the two OpenCV extras: both install a module named
# cv2, and an environment holding both has whichever was installed last.
set -euo pipefail

cd "$(dirname "$0")/.."

passthrough=("$@")

compile_lock() {
    name=$1
    shift
    echo "compiling requirements/$name.txt"
    uv pip compile --universal --generate-hashes --quiet \
        --python-version 3.12 \
        "$@" \
        --output-file "requirements/$name.txt" \
        ${passthrough[@]+"${passthrough[@]}"} \
        pyproject.toml
}

compile_lock headless --extra=headless
compile_lock gui --extra=gui
compile_lock dev --extra=dev --extra=headless
compile_lock dev-gui --extra=dev --extra=gui

echo "done"
