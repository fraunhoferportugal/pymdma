#!/usr/bin/env bash
#
# Ensure shell scripts conform to shellcheck.
#

set -eu

readonly DEBUG=${DEBUG:-unset}
if [ "${DEBUG}" != unset ]; then
    set -x
fi

if ! command -v shellcheck >/dev/null  2>&1; then
    echo 'shellcheck not installed; Please install shellcheck:'
    echo '  sudo apt install -y shellcheck'
    exit 1
fi

shellcheck "$@"
