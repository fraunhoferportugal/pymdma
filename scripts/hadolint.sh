#!/usr/bin/env bash
#
# Haskell Dockerfile Linter
# Dockerfile linter that helps you build best practice Docker images
# https://docs.docker.com/engine/userguide/eng-image/dockerfile_best-practices
#

set -eu

readonly DEBUG=${DEBUG:-unset}
if [ "${DEBUG}" != unset ]; then
    set -x
fi

if ! command -v hadolint >/dev/null  2>&1; then
    echo 'hadolint not installed; Please install hadolint:'
    echo '  download the binary from https://github.com/hadolint/hadolint/releases'
    exit 1
fi

hadolint "$@"
