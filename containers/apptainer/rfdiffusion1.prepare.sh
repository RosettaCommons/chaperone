#!/bin/sh
set -eu
cd "$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
mkdir -p sources/rfdiffusion1/repos/0
if [ -e sources/rfdiffusion1/repos/0/RFdiffusion ]; then
    echo 'Source already exists: sources/rfdiffusion1/repos/0/RFdiffusion; remove or move it before preparing again.' >&2
    exit 1
fi
git clone -- https://github.com/RosettaCommons/RFdiffusion.git sources/rfdiffusion1/repos/0/RFdiffusion
