#!/bin/sh
set -eu
cd "$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
mkdir -p sources/disco/repos/0
if [ -e sources/disco/repos/0/DISCO ]; then
    echo 'Source already exists: sources/disco/repos/0/DISCO; remove or move it before preparing again.' >&2
    exit 1
fi
git clone --branch main -- https://github.com/DISCO-design/DISCO.git sources/disco/repos/0/DISCO
