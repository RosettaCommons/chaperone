#!/bin/sh
set -eu
cd "$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
mkdir -p sources/openfold3/repos/0
if [ -e sources/openfold3/repos/0/openfold-3 ]; then
    echo 'Source already exists: sources/openfold3/repos/0/openfold-3; remove or move it before preparing again.' >&2
    exit 1
fi
git clone -- https://github.com/aqlaboratory/openfold-3.git sources/openfold3/repos/0/openfold-3
