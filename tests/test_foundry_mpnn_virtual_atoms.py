"""RFD3 virtual sidechain slots must not bypass real chemical-name validation."""
from biotite.structure.io.pdbx import CIFBlock, CIFCategory, CIFColumn, CIFFile, MaskValue
import pytest
import numpy as np

from blocks.networks.foundry.mpnn import _stage_rfd3_virtual_atoms


def _fixture(path, complete=True):
    names = ['N','CA','C','O','CB','V0','V8','V9','V1','V2']
    if not complete:
        names[0] = 'OTHER'
    document = CIFFile()
    document['structure'] = CIFBlock()
    document.block['atom_site'] = CIFCategory({
        'group_PDB': ['ATOM']*8+['HETATM','ATOM'],
        'type_symbol': ['N','C','C','O','C','V','V','V','V','V'],
        'label_atom_id': names,
        'label_comp_id': ['UNK']*9+['ALA'],
        'label_asym_id': ['A']*8+['B','C'],
        'label_seq_id': ['1']*10,
        'Cartn_x': [f'{i}.125' for i in range(10)],
        'Cartn_y': ['0.25']*10,
        'Cartn_z': ['0.75']*10,
        'id': [str(i+1) for i in range(10)],
        'pdbx_PDB_model_num': ['1']*10,
        'label_alt_id': CIFColumn(['']*10, [MaskValue.INAPPLICABLE,MaskValue.MISSING]*5),
        'is_motif_atom_with_fixed_seq': ['1']*5+['0']*5,
    })
    document.block['chem_comp_atom'] = CIFCategory({
        'comp_id':['CUSTOM'], 'atom_id':['V1'], 'type_symbol':['V'],
    })
    document.write(path)
    return document


def test_only_documented_unk_virtual_slots_are_removed_without_mutating_input(tmp_path):
    source = tmp_path/'input.cif'
    original = _fixture(source)
    before = source.read_bytes()
    staged = _stage_rfd3_virtual_atoms(source,tmp_path/'sandbox')
    assert staged != source
    assert source.read_bytes() == before
    parsed = CIFFile.read(staged)
    expected = np.array([0,1,2,3,4,7,8,9])
    for name in original.block['atom_site']:
        np.testing.assert_array_equal(parsed.block['atom_site'][name].as_array(str),original.block['atom_site'][name].as_array(str)[expected])
    assert parsed.block['chem_comp_atom'].serialize() == original.block['chem_comp_atom'].serialize()
    # V9, real-ligand V1 and known-residue V2 stay visible to strict validation.
    assert list(parsed.block['atom_site']['label_atom_id'].as_array(str))[-3:]==['V9','V1','V2']


def test_nonpeptide_unk_is_not_sanitized(tmp_path):
    source=tmp_path/'input.cif'
    _fixture(source,complete=False)
    assert _stage_rfd3_virtual_atoms(source,tmp_path/'sandbox') == source


def test_virtual_atom_staging_never_mixes_models(tmp_path):
    source=tmp_path/'models.cif'
    document=_fixture(source)
    site=document.block['atom_site']
    combined={key:CIFColumn(np.tile(site[key].data.array,2),None if site[key].mask is None else np.tile(site[key].mask.array,2)) for key in site}
    combined['pdbx_PDB_model_num']=CIFColumn(['1']*10+['2']*10)
    names=combined['label_atom_id'].data.array.copy()
    names[10]='X'
    combined['label_atom_id']=CIFColumn(names)
    document.block['atom_site']=CIFCategory(combined)
    document.write(source)
    parsed=CIFFile.read(_stage_rfd3_virtual_atoms(source,tmp_path/'sandbox')).block['atom_site']
    assert len(parsed['id'])==18  # only model one's two virtual slots removed
    second=parsed['pdbx_PDB_model_num'].as_array(str)=='2'
    assert {'V0','V8'} <= set(parsed['label_atom_id'].as_array(str)[second])


def test_virtual_atom_staging_rejects_ambiguous_residue_ids(tmp_path):
    source=tmp_path/'ambiguous.cif'
    document=_fixture(source)
    document.block['atom_site']['label_seq_id']=CIFColumn(['.']*10)
    document.write(source)
    with pytest.raises(ValueError,match='unambiguous'):
        _stage_rfd3_virtual_atoms(source,tmp_path/'sandbox')
