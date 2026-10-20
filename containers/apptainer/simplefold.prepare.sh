#!/bin/sh
set -eu
cd "$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
mkdir -p sources/simplefold/repos/0
if [ -e sources/simplefold/repos/0/ml-simplefold ]; then
    echo 'Source already exists: sources/simplefold/repos/0/ml-simplefold; remove or move it before preparing again.' >&2
    exit 1
fi
git clone --branch main -- https://github.com/apple/ml-simplefold.git sources/simplefold/repos/0/ml-simplefold
