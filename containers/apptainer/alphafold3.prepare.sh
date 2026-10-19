#!/bin/sh
set -eu
cd "$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
mkdir -p sources/alphafold3/repos/0
if [ -e sources/alphafold3/repos/0/alphafold3 ]; then
    echo 'Source already exists: sources/alphafold3/repos/0/alphafold3; remove or move it before preparing again.' >&2
    exit 1
fi
git clone --branch main -- https://github.com/google-deepmind/alphafold3 sources/alphafold3/repos/0/alphafold3
