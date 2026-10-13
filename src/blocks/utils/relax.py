import logging

from blocks.constants import TIP_BY_RESTYPE
from chaperone import PipelineDataHandler
from chaperone.block_config.inline import InlineBlockFunction
from chaperone.data.schema import DesignMetadata
from pathlib import Path

logger = logging.getLogger(__name__)


class PyrosettaRotamerRelaxBlock(InlineBlockFunction):
    """PyRosetta rotamer relaxation for sidechain optimization."""

    _name = "rotamer_relax"
    _display_name = "Rotamer Relax"
    _description = "PyRosetta rotamer relaxation"
    _inline = True
    _chunked = True

    def execute(
        self,
        handler: PipelineDataHandler,
        sandbox_dir: Path,
        chunk_id: int,
    ) -> "PipelineDataHandler":
        """Run rotamer relaxation inline using PyRosetta."""

        from blocks.utils.pyrosetta import init_pyrosetta, atom_array_to_pose

        init_pyrosetta()

        outs = []
        for loaded in handler.get_data(which="last", load=True):
            atom_array = loaded.atom_array
            pose = atom_array_to_pose(atom_array)

            # Perform rotamer relaxation
            resno_to_min = [
                pose.pdb_info().pdb2pose(idx[0], int(idx[1:]))
                for idx in loaded.metadata[DesignMetadata.MOTIF]
            ]
            if any(resno == 0 for resno in resno_to_min):
                logger.error("Residue number mapping failed!")
                continue

            min_sc_fix_tip_fix_bb(pose, resno_to_min, unfreeze_bb_in_loops=True)

            # Write out relaxed structure
            out_path = sandbox_dir / f"{loaded.basename}_relaxed.cif"
            pose.dump_cif(out_path)
            outs.append(loaded.to_row_with_updates(name=self.name, path=out_path))

        return PipelineDataHandler(outs)


def min_sc_fix_tip_fix_bb(pose, resnos, unfreeze_bb_in_loops=True):
    """
    Minimize selected atoms of given catalytic residues.

    Unfreezes hydrogens and non-tip / non-backbone atoms.
    Also unfreezes backbone atoms if residue is in a loop (secstruct == 'L') if specified.
    """
    import pyrosetta as pyr
    import pyrosetta.rosetta

    dssp = pyr.rosetta.core.scoring.dssp.Dssp(pose)
    secstruct = dssp.get_dssp_secstruct()

    movemap = pyr.MoveMap()
    movemap.set_bb(False)
    movemap.set_chi(False)
    movemap.set_jump(False)

    for resno in resnos:
        res = pose.residue(resno)
        restype = res.name3()
        tip_atoms = TIP_BY_RESTYPE.get(restype, None)
        context = secstruct[resno - 1]

        # iterate every atom in the residue
        for i in range(1, res.natoms() + 1):
            if res.atom_is_hydrogen(i):
                # Unfreeze all hydrogens
                movemap.set_atom(pyr.rosetta.core.id.AtomID(i, resno), True)
            elif res.atom_is_backbone(i):
                if context == "L" and unfreeze_bb_in_loops:
                    # If in loop, unfreeze backbone atoms
                    movemap.set_atom(pyr.rosetta.core.id.AtomID(i, resno), True)
            elif tip_atoms is not None and res.atom_name(i).strip() not in tip_atoms:
                # Unfreeze all non-tip atoms (that are not hydrogens or backbone)
                movemap.set_atom(pyr.rosetta.core.id.AtomID(i, resno), True)

    cart_scorefxn = pyr.create_score_function("beta_nov16_cart.wts")
    cart_scorefxn.set_weight(pyr.rosetta.core.scoring.cart_bonded, 5.0)

    min_mover = pyr.rosetta.protocols.minimization_packing.MinMover()
    min_mover.score_function(cart_scorefxn)
    min_mover.movemap(movemap)
    min_mover.min_type("lbfgs_armijo")
    min_mover.cartesian(True)
    min_mover.max_iter(1000)
    min_mover.tolerance(0.0001)

    # perform minimization
    min_mover.apply(pose)
