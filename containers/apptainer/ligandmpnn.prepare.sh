#!/bin/sh
set -eu
cd "$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
mkdir -p sources/ligandmpnn/repos/0
if [ -e sources/ligandmpnn/repos/0/LigandMPNN ]; then
    echo 'Source already exists: sources/ligandmpnn/repos/0/LigandMPNN; remove or move it before preparing again.' >&2
    exit 1
fi
git clone -- https://github.com/dauparas/LigandMPNN sources/ligandmpnn/repos/0/LigandMPNN
