#!/bin/sh
set -eu
cd "$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
mkdir -p sources/bindcraft/repos/0
if [ -e sources/bindcraft/repos/0/BindCraft ]; then
    echo 'Source already exists: sources/bindcraft/repos/0/BindCraft; remove or move it before preparing again.' >&2
    exit 1
fi
git clone -- https://github.com/martinpacesa/BindCraft.git sources/bindcraft/repos/0/BindCraft
