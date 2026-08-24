#!/usr/bin/env bash
set -Eeuo pipefail

# Configuration and legacy aliases are resolved once by the packaged CLI.
exec thtts "$@"
