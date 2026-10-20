#!/bin/sh
set -eu
cd "$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
mkdir -p sources/metal3d/repos/0
if [ -e sources/metal3d/repos/0/metal-site-prediction ]; then
    echo 'Source already exists: sources/metal3d/repos/0/metal-site-prediction; remove or move it before preparing again.' >&2
    exit 1
fi
git clone -- https://github.com/lcbc-epfl/metal-site-prediction.git sources/metal3d/repos/0/metal-site-prediction
