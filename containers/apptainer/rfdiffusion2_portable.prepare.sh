#!/bin/sh
set -eu
cd "$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
mkdir -p sources/rfdiffusion2_portable/repos/0
if [ -e sources/rfdiffusion2_portable/repos/0/RFdiffusion2 ]; then
    echo 'Source already exists: sources/rfdiffusion2_portable/repos/0/RFdiffusion2; remove or move it before preparing again.' >&2
    exit 1
fi
git clone -- https://github.com/RosettaCommons/RFdiffusion2 sources/rfdiffusion2_portable/repos/0/RFdiffusion2
