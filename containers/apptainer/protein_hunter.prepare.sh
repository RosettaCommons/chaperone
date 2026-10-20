#!/bin/sh
set -eu
cd "$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
mkdir -p sources/protein_hunter/repos/0
if [ -e sources/protein_hunter/repos/0/Protein-Hunter ]; then
    echo 'Source already exists: sources/protein_hunter/repos/0/Protein-Hunter; remove or move it before preparing again.' >&2
    exit 1
fi
git clone -- https://github.com/yehlincho/Protein-Hunter.git sources/protein_hunter/repos/0/Protein-Hunter
