# Complementaries to Bcov & his phd thesis: https://github.com/bcov77/py_contact_ms/blob/main/py_contact_ms.py
#!/usr/bin/env python
import math

import numpy as np
from scipy.spatial.distance import cdist

"""
API Notes:

The only two functions you really need to call are get_radii_from_names() and calculate_contact_ms() 

"""


def calculate_contact_ms(
    binder_xyz, binder_radii, target_xyz, target_radii, return_calc=False
):
    """
    Main entrypoint into the code

    Calculate contact molecular surface of your binder against the target

    Contact molecular surface has units of A^2 and is a distance weighted surface area of your target

    Do not provide your own radii, you need to use the very specific radii raturned from get_radii_from_names()

    Parameters
    -------
    binder_xyz   : np.ndarray (N0, 3)
    binder_radii : np.ndarray (N0,)
    target_xyz   : np.ndarray (N1, 3)
    target_radii : np.ndarray (N1,)
    return_calc  : bool


    Returns
    -------
    cms                 : float -- The contact molecular surface
    per_atom_target_cms : np.ndarray shape (N1,) — The per-atom contact_ms of your target molecule
    """

    calc = MolecularSurfaceCalculator()
    calc.add_binder_and_target(binder_xyz, binder_radii, target_xyz, target_radii)
    cms, per_atom_target_cms = calc.CalcLoaded()

    if return_calc:
        return cms, per_atom_target_cms, calc
    else:
        return cms, per_atom_target_cms


def get_radii_from_names(res_names, atom_names):
    """
    Look up radii for parallel lists of residue names and atom names.

    Residue and atom names should be stripped (So 'N' instead of ' N  ')

    Parameters
    ----------
    res_names  : list[str]  length N
    atom_names : list[str]  length N

    Returns
    -------
    radii : np.ndarray shape (N,) — 0.0 for any unmatched atom
    """
    radii_ = read_sc_radii()
    radii = np.zeros(len(res_names), dtype=np.float64)
    for i, (res, atom) in enumerate(zip(res_names, atom_names)):
        for radius_obj in radii_:
            if not wildcard_match(res, radius_obj.residue, len(res) + 2):
                continue
            if not wildcard_match(atom, radius_obj.atom, len(atom) + 2):
                continue
            radii[i] = radius_obj.radius
            break
    return radii


def partition_pose(pose, jump_id=1):
    """
    If for some reason you are calling these functions with a pyrosetta pose...

    Partition a pose by jump and return xyz and radii arrays for each molecule.

    Calls extract_atom_data_from_pose to get per-atom residue/atom names and
    coordinates, then get_radii_from_names to look up radii.  Atoms with no
    matching radius (radius == 0) are filtered out.

    Returns
    -------
    xyz_0   : np.ndarray (N0, 3)
    radii_0 : np.ndarray (N0,)
    xyz_1   : np.ndarray (N1, 3)
    radii_1 : np.ndarray (N1,)
    """
    rn0, an0, xyz_0, rn1, an1, xyz_1 = extract_atom_data_from_pose(pose, jump_id)

    radii_0 = get_radii_from_names(rn0, an0)
    radii_1 = get_radii_from_names(rn1, an1)

    mask_0 = radii_0 > 0
    mask_1 = radii_1 > 0

    return xyz_0[mask_0], radii_0[mask_0], xyz_1[mask_1], radii_1[mask_1]


SC_RADII_LIB = """
ALA  CB      1.95
ARG  NH*     1.70
ARG  CZ      1.80
ARG  NE      1.65
ARG  CD      1.90
ARG  CG      1.90
ASN  ND2     1.70
ASN  OD1     1.60
ASN  CG      1.80
ASP  OD*     1.60
ASP  CG      1.80
GLN  NE2     1.70
GLN  OE1     1.60
GLN  CD      1.80
GLN  CG      1.90
GLU  OE*     1.60
GLU  CD      1.80
GLU  CG      1.90
GLY  CA      1.90
HIS  CD2     1.90
HIS  NE2     1.65
HIS  CE1     1.90
HIS  ND1     1.65
HIS  CG      1.80
HOH  O*      1.70
ILE  CD1     1.95
ILE  CG1     1.90
ILE  CB      1.85
ILE  CG2     1.95
LEU  CD*     1.95

LEU  CG      1.85
LYS  NZ      1.75
LYS  CE      1.90
LYS  CD      1.90
LYS  CG      1.90
MET  CE      1.95
MET  CG      1.90
PHE  CD*     1.90
PHE  CE*     1.90
PHE  CZ      1.90
PHE  CG      1.80
PRO  CD      1.90
PRO  CG      1.90
SER  OG      1.70
SUL  S       1.90
SUL  O*      1.65
THR  CG2     1.95
THR  OG1     1.70
THR  CB      1.85
TRP  CE2     1.80
TRP  CE3     1.90
TRP  CD1     1.90
TRP  CD2     1.80
TRP  CZ*     1.90
TRP  CH2     1.90
TRP  NE1     1.65
TRP  CG      1.80
TYR  OH      1.70
TYR  CD*     1.90
TYR  CE*     1.90
TYR  CZ      1.80
TYR  CG      1.80
VAL  CG*     1.95
VAL  CB      1.85
WAT  O       1.70
WAT  O*      1.70
LG*  HARO    1.00
LG*  AROC    2.00
LG*  ARO     2.00
LG*  NHIS    1.75
LG*  NHI     1.75
LG*  COO     1.55
LG*  OCBB    1.55
LG*  OCB     1.55
LG*  OOC     1.55
LG*  NLYS    1.75
*    H       0.50
*    H*      0.50
*    1H*     0.50
*    2H*     0.50
*    3H*     0.50
*    C       1.80
*    CA      1.85
*    CB      1.90
*    C*      1.85
*    N       1.65
*    N*      1.65
*    O       1.60
*    O*      1.60
*    OXT     1.60
*    OT*     1.60
*    S*      1.90
*    P*      2.15
*    F*      1.50
"""


class ResultsSurface:
    def __init__(self):
        self.d_mean = 0.0
        self.d_median = 0.0
        self.s_mean = 0.0
        self.s_median = 0.0
        self.nAtoms = 0
        self.nBuriedAtoms = 0
        self.nBlockedAtoms = 0
        self.nAllDots = 0
        self.nTrimmedDots = 0
        self.nBuriedDots = 0
        self.nAccessibleDots = 0
        self.trimmedArea = 0.0


class ResultsDots:
    def __init__(self):
        self.convex = 0
        self.concave = 0
        self.toroidal = 0


class RESULTS:
    def __init__(self):
        self.sc = 0.0
        self.area = 0.0
        self.distance = 0.0
        self.perimeter = 0.0
        self.nAtoms = 0
        self.surface = [ResultsSurface(), ResultsSurface(), ResultsSurface()]
        self.dots = ResultsDots()
        self.valid = 0


class AtomArray:
    """
    Struct-of-arrays container for Atom data.

    Grows dynamically via append() or extend_from_arrays(); call finalize() to
    trim arrays to their true size before running vectorised operations.

    Indexing with a slice, boolean mask, or integer array returns a new
    AtomArray containing the selected rows.
    """

    _INITIAL_CAP = 256
    _FLOAT_FIELDS = ("radius", "density")
    _INT8_FIELDS = ("molecule", "atten", "access")
    _INT32_FIELDS = ("natom", "nresidue")

    def __init__(self):
        cap = self._INITIAL_CAP
        self._n = 0
        self._cap = cap

        for f in self._FLOAT_FIELDS:
            setattr(self, f, np.zeros(cap, dtype=np.float64))
        for f in self._INT8_FIELDS:
            setattr(self, f, np.zeros(cap, dtype=np.int8))
        for f in self._INT32_FIELDS:
            setattr(self, f, np.zeros(cap, dtype=np.int32))

        self.atom_name = [""] * cap
        self.residue_name = [""] * cap
        self.xyz = np.zeros((cap, 3), dtype=np.float64)

    def _grow(self):
        new_cap = self._cap * 2
        for f in self._FLOAT_FIELDS + self._INT8_FIELDS + self._INT32_FIELDS:
            old = getattr(self, f)
            new = np.zeros(new_cap, dtype=old.dtype)
            new[: self._n] = old[: self._n]
            setattr(self, f, new)
        self.atom_name = self.atom_name + [""] * self._cap
        self.residue_name = self.residue_name + [""] * self._cap
        xyz_old = self.xyz
        self.xyz = np.zeros((new_cap, 3), dtype=np.float64)
        self.xyz[: self._n] = xyz_old[: self._n]
        self._cap = new_cap

    def append(self, atom):
        """Copy data from an Atom into the array."""
        if self._n >= self._cap:
            self._grow()
        i = self._n
        self.xyz[i, 0] = atom.x_
        self.xyz[i, 1] = atom.y_
        self.xyz[i, 2] = atom.z_
        self.radius[i] = atom.radius
        self.density[i] = atom.density
        self.natom[i] = atom.natom
        self.nresidue[i] = atom.nresidue
        self.molecule[i] = atom.molecule
        self.atten[i] = atom.atten
        self.access[i] = atom.access
        self.atom_name[i] = atom.atom
        self.residue_name[i] = atom.residue
        self._n += 1

    def extend_from_arrays(self, xyz, radii, density, molecule):
        """
        Bulk-append N atoms from pre-computed numpy arrays.

        natom values are assigned as sequential indices starting from the
        current length.  nresidue, atom_name, and residue_name are left at
        their zero/empty defaults.

        Parameters
        ----------
        xyz      : np.ndarray (N, 3)
        radii    : np.ndarray (N,)
        density  : float or np.ndarray broadcastable to (N,)
        molecule : int  0 or 1
        """
        n = len(radii)
        if n == 0:
            return
        while self._n + n > self._cap:
            self._grow()
        i = self._n
        self.xyz[i : i + n] = xyz
        self.radius[i : i + n] = radii
        self.density[i : i + n] = density
        self.molecule[i : i + n] = molecule
        self.natom[i : i + n] = np.arange(i, i + n, dtype=np.int32)
        self.access[i : i + n] = 0
        self._n += n

    def __len__(self):
        return self._n

    def __bool__(self):
        return self._n > 0

    def __getitem__(self, idx):
        """Slice, boolean mask, or integer array → new AtomArray."""

        # Convert slice to explicit indices
        if isinstance(idx, slice):
            indices = np.arange(self._n)[idx]
        else:
            indices = np.asarray(idx)

            if indices.dtype == bool:
                if indices.shape[0] != self._n:
                    raise IndexError("Boolean index wrong length")
                indices = np.nonzero(indices)[0]

        # Handle negative fancy indices
        indices = np.where(indices < 0, indices + self._n, indices)

        if np.any((indices < 0) | (indices >= self._n)):
            raise IndexError("AtomArray index out of range")

        # Build new AtomArray
        new = AtomArray()

        new._n = len(indices)
        new._cap = new._n

        for f in self._FLOAT_FIELDS + self._INT8_FIELDS + self._INT32_FIELDS:
            arr = getattr(self, f)
            setattr(new, f, arr[indices].copy())

        new.xyz = self.xyz[indices].copy()
        new.atom_name = [self.atom_name[i] for i in indices]
        new.residue_name = [self.residue_name[i] for i in indices]

        new.finalize()
        return new

    def finalize(self):
        """Trim all arrays to [0:n].  Call once after all appends are done."""
        for f in self._FLOAT_FIELDS + self._INT8_FIELDS + self._INT32_FIELDS:
            setattr(self, f, getattr(self, f)[: self._n].copy())
        self.xyz = self.xyz[: self._n]
        self.atom_name = self.atom_name[: self._n]
        self.residue_name = self.residue_name[: self._n]


class DotArray:
    """
    Struct-of-arrays container for DOT data.

    No view class: the caller writes individual field values via append() and
    reads via the numpy arrays directly (enabling vectorised operations).
    """

    _INITIAL_CAP = 4096

    def __init__(self):
        cap = self._INITIAL_CAP
        self._n = 0
        self._cap = cap

        self.coor_xyz = np.zeros((cap, 3), dtype=np.float64)
        self.outnml_xyz = np.zeros((cap, 3), dtype=np.float64)
        self.area = np.zeros(cap, dtype=np.float64)
        for f in ("buried", "type_"):
            setattr(self, f, np.zeros(cap, dtype=np.int8))
        self.atom_idx = np.zeros(cap, dtype=np.int32)

    def _grow(self):
        new_cap = self._cap * 2
        for f in ("coor_xyz", "outnml_xyz", "area", "buried", "type_", "atom_idx"):
            old = getattr(self, f)
            new = np.zeros((new_cap,) + old.shape[1:], dtype=old.dtype)
            new[: self._n] = old[: self._n]
            setattr(self, f, new)
        self._cap = new_cap

    def append(self, coor_xyz, outnml_xyz, area, buried, type_, atom_idx):
        if self._n >= self._cap:
            self._grow()
        i = self._n
        self.coor_xyz[i] = coor_xyz
        self.outnml_xyz[i] = outnml_xyz
        self.area[i] = area
        self.buried[i] = buried
        self.type_[i] = type_
        self.atom_idx[i] = atom_idx
        self._n += 1

    def extend(self, coor_xyz, outnml_xyz, area, buried, type_, atom_idx):
        """Batch-append n dots in a single slice assignment."""
        n = len(coor_xyz)
        if n == 0:
            return
        while self._n + n > self._cap:
            self._grow()
        i = self._n
        self.coor_xyz[i : i + n] = coor_xyz
        self.outnml_xyz[i : i + n] = outnml_xyz
        self.area[i : i + n] = area
        self.buried[i : i + n] = buried
        self.type_[i : i + n] = type_
        self.atom_idx[i : i + n] = atom_idx
        self._n += n

    def __len__(self):
        return self._n

    def __bool__(self):
        return self._n > 0

    def finalize(self):
        for f in ("coor_xyz", "outnml_xyz", "area", "buried", "type_", "atom_idx"):
            setattr(self, f, getattr(self, f)[: self._n].copy())


class ProbeArray:
    """Struct-of-arrays container for PROBE data."""

    _INITIAL_CAP = 1024

    def __init__(self):
        cap = self._INITIAL_CAP
        self._n = 0
        self._cap = cap

        self.height = np.zeros(cap, dtype=np.float64)
        self.atom_idx_0 = np.zeros(cap, dtype=np.int32)
        self.atom_idx_1 = np.zeros(cap, dtype=np.int32)
        self.atom_idx_2 = np.zeros(cap, dtype=np.int32)
        self.point_xyz = np.zeros((cap, 3), dtype=np.float64)
        self.alt_xyz = np.zeros((cap, 3), dtype=np.float64)

    def _grow(self):
        new_cap = self._cap * 2
        for f in ("height", "atom_idx_0", "atom_idx_1", "atom_idx_2"):
            old = getattr(self, f)
            new = np.zeros(new_cap, dtype=old.dtype)
            new[: self._n] = old[: self._n]
            setattr(self, f, new)

        point_xyz_old = self.point_xyz
        self.point_xyz = np.zeros((new_cap, 3), dtype=np.float64)
        self.point_xyz[: self._n] = point_xyz_old[: self._n]

        alt_xyz_old = self.alt_xyz
        self.alt_xyz = np.zeros((new_cap, 3), dtype=np.float64)
        self.alt_xyz[: self._n] = alt_xyz_old[: self._n]

        self._cap = new_cap

    def extend_from_arrays(
        self, atom_idx_0, atom_idx_1, atom_idx_2, height, point_xyz, alt_xyz
    ):
        """
        Bulk-append N probes from pre-computed numpy arrays.

        Parameters
        ----------
        atom_idx_0, atom_idx_1, atom_idx_2 : array-like (N,)   int32
        height    : array-like (N,)    float64
        point_xyz : array-like (N, 3)  float64  probe centre coordinates
        alt_xyz   : array-like (N, 3)  float64  alternate axis coordinates
        """
        n = len(height)
        if n == 0:
            return
        while self._n + n > self._cap:
            self._grow()
        i = self._n
        self.atom_idx_0[i : i + n] = atom_idx_0
        self.atom_idx_1[i : i + n] = atom_idx_1
        self.atom_idx_2[i : i + n] = atom_idx_2
        self.height[i : i + n] = height
        self.point_xyz[i : i + n] = point_xyz
        self.alt_xyz[i : i + n] = alt_xyz
        self._n += n

    def __len__(self):
        return self._n

    def __bool__(self):
        return self._n > 0

    def finalize(self):
        for f in (
            "height",
            "point_xyz",
            "alt_xyz",
            "atom_idx_0",
            "atom_idx_1",
            "atom_idx_2",
        ):
            setattr(self, f, getattr(self, f)[: self._n].copy())


class SimpleNeighborArray:
    """
    Store the information needed to work with neighbors
    """

    def __init__(self, cap, max_neighbors):
        self._cap = cap
        self._max_neighbors = max_neighbors

        self.xyz = np.full((cap, max_neighbors, 3), np.nan, dtype=np.float64)
        self.radius = np.full((cap, max_neighbors), np.nan, dtype=np.float64)
        self.natom = np.full((cap, max_neighbors), -1, dtype=np.int32)
        self.nneighbors = np.zeros((cap,), dtype=np.int32)

    def __len__(self):
        return self._cap

    def __bool__(self):
        return self._cap > 0


ATTEN_BLOCKER = 1
ATTEN_2 = 2
ATTEN_BURIED_FLAGGED = 5
ATTEN_6 = 6

MAX_SUBDIV = 100
PI = math.pi


def extract_atom_data_from_pose(pose, jump_id=1):
    """
    Extract per-atom residue names, atom names, and xyz coordinates from a
    PyRosetta pose, partitioned by jump into two molecules.

    Returns
    -------
    res_names_0  : list[str]         residue 3-letter names for molecule 0
    atom_names_0 : list[str]         stripped atom names for molecule 0
    xyz_0        : np.ndarray (N0,3) coordinates for molecule 0
    res_names_1  : list[str]
    atom_names_1 : list[str]
    xyz_1        : np.ndarray (N1,3)
    """
    from pyrosetta import rosetta

    if jump_id > pose.num_jump():
        raise ValueError("Jump ID out of bounds")

    is_upstream = rosetta.utility.vector1_bool(pose.size())
    if jump_id > 0:
        is_upstream = pose.fold_tree().partition_by_jump(jump_id)
    else:
        for i in range(1, pose.size() + 1):
            is_upstream[i] = True

    res_names = [[], []]
    atom_names = [[], []]
    xyzs = [[], []]

    for i in range(1, pose.size() + 1):
        residue = pose.residue(i)

        if residue.type().name() == "VRT":
            continue
        if residue.type().is_metal():
            continue

        mol = 0 if is_upstream[i] else 1

        for j in range(1, residue.nheavyatoms() + 1):
            if residue.is_virtual(j):
                continue

            xyz = residue.xyz(j)
            res_names[mol].append(residue.name3())
            atom_names[mol].append(residue.atom_name(j).strip())
            xyzs[mol].append([xyz.x, xyz.y, xyz.z])

    xyz_arrays = [np.array(xyzs[m], dtype=np.float64).reshape(-1, 3) for m in range(2)]

    return (
        res_names[0],
        atom_names[0],
        xyz_arrays[0],
        res_names[1],
        atom_names[1],
        xyz_arrays[1],
    )


def read_sc_radii():
    """
    Read side-chain radii definitions.

    Returns:
        1 if radii were successfully read (non-empty)
        0 otherwise
    """

    radii = []

    for line in SC_RADII_LIB.split("\n"):
        line = line.strip()
        if not line:
            continue

        parts = line.split()
        if len(parts) < 3:
            continue

        residue, atom, radius_value = parts[0], parts[1], parts[2]

        try:
            radius_float = float(radius_value)
        except ValueError:
            continue

        if residue and atom and radius_float > 0:
            # Create ATOM_RADIUS-like object
            radius_obj = type("ATOM_RADIUS", (), {})()
            radius_obj.residue = residue
            radius_obj.atom = atom
            radius_obj.radius = radius_float

            radii.append(radius_obj)

    return radii


def wildcard_match(query: str, pattern: str, l: int):
    """
    Inline residue and atom name matching.
    Mirrors C++ logic exactly.
    """

    qi = 0
    pi = 0

    while True:
        l -= 1
        if l <= 0:
            break

        q = query[qi] if qi < len(query) else "\0"
        p = pattern[pi] if pi < len(pattern) else "\0"

        match = (q == p) or (q != "\0" and p == "*") or (q == " " and p == "\0")

        if not match:
            return 0

        # Allow anything following a * in pattern
        if p == "*" and (pi + 1 >= len(pattern) or pattern[pi + 1] == "\0"):
            return 1

        if q != "\0":
            qi += 1
        if p != "\0":
            pi += 1

    return 1


class MolecularSurfaceCalculator:
    def __init__(self):
        self.settings = type("Settings", (), {})()
        self.settings.rp = 1.7
        self.settings.density = 15.0
        self.settings.band = 1.5
        self.settings.sep = 8.0
        self.settings.weight = 0.5
        self.settings.binwidth_dist = 0.02
        self.settings.binwidth_norm = 0.02

        self.reset()

    def reset(self):
        self.run = type("Run", (), {})()
        self.run.radmax = 0.0
        self.run.results = RESULTS()
        self.run.atoms = AtomArray()
        self.run.dots = [DotArray(), DotArray()]
        self.run.trimmed_dots = [DotArray(), DotArray()]
        self.run.probes = ProbeArray()
        self.run.neighbor_array = None
        self.run.buried_array = None
        self.run.toroid_queue = []

    def CalcLoaded(self):
        self.run.results.valid = 0
        assert len(self.run.atoms) > 0

        # Trim atom arrays to true size before vectorised attention assignment
        self.run.atoms.finalize()

        self.assign_attention_numbers(self.run.atoms)

        self.generate_molecular_surfaces()

        # Trim dot / probe arrays after surface generation
        self.run.dots[0].finalize()
        self.run.dots[1].finalize()
        self.run.probes.finalize()

        cms_return = self.calc_contact_molecular_surface(target_side=True)

        return cms_return

    def generate_molecular_surfaces(self):
        assert len(self.run.atoms) > 0

        self.calc_dots_for_all_atoms(self.run.atoms)

    def add_binder_and_target(self, binder_xyz, binder_radii, target_xyz, target_radii):
        self.reset()
        self.AddMolecule(0, binder_xyz, binder_radii)
        self.AddMolecule(1, target_xyz, target_radii)

    def AddMolecule(self, molecule, xyz, radii):
        """
        Load atoms for one molecule from numpy arrays.

        Parameters
        ----------
        molecule : int          0 or 1
        xyz      : np.ndarray   shape (N, 3) — atom coordinates
        radii    : np.ndarray   shape (N,)   — atom radii; atoms with radius
                                               <= 0 are skipped
        """
        mol_val = 1 if molecule == 1 else 0
        mask = np.asarray(radii) > 0
        xyz_f = np.asarray(xyz)[mask]
        radii_f = np.asarray(radii)[mask]
        n = len(radii_f)
        self.run.atoms.extend_from_arrays(
            xyz_f, radii_f, self.settings.density, mol_val
        )
        self.run.results.surface[mol_val].nAtoms += n
        self.run.results.nAtoms += n

    def calc_contact_molecular_surface(self, target_side=True):
        """
        Compute the contact molecular surface (vectorised).

        For each buried dot on the *query* molecule the nearest buried dot on
        the *reference* molecule is found; the weighted area sum gives the CMS.

        target_side=True  (default): query=mol1, reference=mol0.
            Returns per-atom values sized (n_mol1,).
        target_side=False           : query=mol0, reference=mol1.
            Returns per-atom values sized (n_mol0,).

        Returns
        -------
        total_cms    : float
        per_atom_cms : np.ndarray shape (n_query_atoms,)
        """
        n = len(self.run.atoms)
        mol = self.run.atoms.molecule[:n]
        n_mol1 = int((mol == 1).sum())
        n_mol0 = n - n_mol1

        if target_side:
            dots_q, dots_r = self.run.dots[1], self.run.dots[0]
            n_q = n_mol1
            atom_offset = n_mol0  # mol1 atoms start here in the global array
        else:
            dots_q, dots_r = self.run.dots[0], self.run.dots[1]
            n_q = n_mol0
            atom_offset = 0  # mol0 atoms start at index 0

        zero_ret = (0.0, np.zeros(n_q, dtype=np.float64))

        if len(dots_r) == 0:
            return zero_ret

        buried_q = dots_q.buried.astype(bool)
        buried_r = dots_r.buried.astype(bool)

        if not buried_q.any() or not buried_r.any():
            return zero_ret

        xyz_q = dots_q.coor_xyz[buried_q]  # (Kq, 3)
        xyz_r = dots_r.coor_xyz[buried_r]  # (Kr, 3)
        area_q = dots_q.area[buried_q]  # (Kq,)

        dist_sq = cdist(xyz_r, xyz_q, metric="sqeuclidean")  # (Kr, Kq)
        min_dist_sq = dist_sq.min(axis=0)  # (Kq,)

        per_dot_cms = area_q * np.exp(-min_dist_sq * self.settings.weight)  # (Kq,)

        # Sum per-dot contributions into per-atom bins using a 0-based local index.
        local_idx = dots_q.atom_idx[buried_q] - atom_offset  # (Kq,)
        per_atom_cms = np.bincount(
            local_idx, weights=per_dot_cms, minlength=n_q
        ).astype(np.float64)

        total_cms = float(per_atom_cms.sum())
        return total_cms, per_atom_cms

    def assign_attention_numbers(self, atoms, all_atoms=False):
        """
        Assign attention values to all atoms (vectorised).

        Replaces the O(N²) Python double-loop with a single scipy cdist call
        to compute all inter-molecule pairwise distances at once.
        """
        n = len(atoms)
        mol = atoms.molecule[:n]

        if all_atoms:
            atoms.atten[:n] = ATTEN_BURIED_FLAGGED
            for m in range(2):
                self.run.results.surface[m].nBuriedAtoms += int((mol == m).sum())
            return 1

        xyz = atoms.xyz[:n]
        mask0 = mol == 0
        mask1 = mol == 1
        xyz0 = xyz[mask0]  # (N0, 3)
        xyz1 = xyz[mask1]  # (N1, 3)

        if len(xyz0) > 0 and len(xyz1) > 0:
            d = cdist(xyz0, xyz1)  # (N0, N1)
            min0 = d.min(axis=1)  # min distance for each mol-0 atom
            min1 = d.min(axis=0)  # min distance for each mol-1 atom
        else:
            min0 = np.full(mask0.sum(), 99999.0)
            min1 = np.full(mask1.sum(), 99999.0)

        idx0 = np.where(mask0)[0]
        idx1 = np.where(mask1)[0]
        blocker0 = min0 >= self.settings.sep
        blocker1 = min1 >= self.settings.sep

        atoms.atten[idx0[blocker0]] = ATTEN_BLOCKER
        atoms.atten[idx0[~blocker0]] = ATTEN_BURIED_FLAGGED
        atoms.atten[idx1[blocker1]] = ATTEN_BLOCKER
        atoms.atten[idx1[~blocker1]] = ATTEN_BURIED_FLAGGED

        self.run.results.surface[0].nBlockedAtoms += int(blocker0.sum())
        self.run.results.surface[0].nBuriedAtoms += int((~blocker0).sum())
        self.run.results.surface[1].nBlockedAtoms += int(blocker1.sum())
        self.run.results.surface[1].nBuriedAtoms += int((~blocker1).sum())

        return 1

    def calc_dots_for_all_atoms(self, _atoms_unused):
        """
        Main surface generation loop.
        """

        # Compute maximum atom radius
        self.run.radmax = self.run.atoms.radius.max()

        good_atom = self.build_neighbor_arrays()

        # Run second_loop for all good atoms at once
        good_indices = np.where(good_atom)[0]
        self.second_loop(self.run.atoms[good_indices])

        # Build convex_queue (vectorised filter over good atoms)
        atten = self.run.atoms.atten[good_indices]
        access = self.run.atoms.access[good_indices].astype(bool)
        n_buried = self.run.buried_array.nneighbors[good_indices]

        convex_mask = (
            access & (atten > ATTEN_BLOCKER) & ~((atten == ATTEN_6) & (n_buried == 0))
        )
        convex_queue = list(good_indices[convex_mask])

        self.generate_convex_surface(self.run.atoms[convex_queue])

        self.generate_toroidal_surfaces()

        self.run.probes.finalize()
        # Concave surface
        if self.settings.rp > 0:
            self.generate_concave_surface()

        return 1

    def generate_toroidal_surfaces(self):
        natoms1 = [x[0] for x in self.run.toroid_queue]
        natoms2 = [x[1] for x in self.run.toroid_queue]
        uijs = np.array([x[2] for x in self.run.toroid_queue])
        tijs = np.array([x[3] for x in self.run.toroid_queue])
        rijs = np.array([x[4] for x in self.run.toroid_queue])
        betweens = np.array([x[5] for x in self.run.toroid_queue])

        atom1_has_access, atom2_has_access = self.generate_toroidal_surface(
            self.run.atoms[natoms1], self.run.atoms[natoms2], uijs, tijs, rijs, betweens
        )
        self.run.atoms.access[natoms1] |= atom1_has_access
        self.run.atoms.access[natoms2] |= atom2_has_access

    def build_neighbor_arrays(self):
        """
        Vectorised replacement for the find_neighbors_for_atom loop +
        generate_neighbor_array.  Computes all-pairs squared distances once
        with cdist, derives neighbor / buried masks with numpy broadcasting,
        then fills neighbor_array and buried_array in a single O(N) Python
        pass with per-atom numpy indexing.

        Returns
        -------
        good_atom : bool ndarray, shape (N,)
        """
        atoms = self.run.atoms
        N = len(atoms)
        rp = self.settings.rp

        xyz = atoms.xyz[:N]  # (N, 3)
        radius = atoms.radius[:N]  # (N,)
        mol = atoms.molecule[:N]  # (N,)
        atten = atoms.atten[:N]  # (N,)
        natom = atoms.natom[:N]  # (N,)

        # ── all-pairs squared distances ───────────────────────────────────
        d2 = cdist(xyz, xyz, metric="sqeuclidean")  # (N, N)

        # ── bridge threshold matrix ───────────────────────────────────────
        r_sum = radius[:, None] + radius[None, :]  # (N, N)
        bridge2 = (r_sum + 2.0 * rp) ** 2  # (N, N)

        # ── boolean context masks ─────────────────────────────────────────
        active = atten > 0  # (N,)
        same_mol = mol[:, None] == mol[None, :]  # (N, N)
        same_mol_off = same_mol & ~np.eye(N, dtype=bool)  # exclude self

        # ── coincident atoms check ────────────────────────────────────────
        coincident = same_mol_off & active[:, None] & active[None, :] & (d2 <= 0.0001)
        if coincident.any():
            i0, j0 = map(int, np.argwhere(coincident)[0])
            raise RuntimeError(
                f"Coincident atoms: "
                f"{natom[i0]}:{atoms.residue_name[i0]}:{atoms.atom_name[i0]} == "
                f"{natom[j0]}:{atoms.residue_name[j0]}:{atoms.atom_name[j0]}"
            )

        # ── neighbor mask (same molecule, within bridge) ──────────────────
        neighbor_mask = (
            active[:, None] & active[None, :] & same_mol_off & (d2 < bridge2)
        )

        # ── cross-molecule masks ──────────────────────────────────────────
        diff_mol = ~same_mol  # (N, N); diag=False
        buried_eligible = atten >= ATTEN_BURIED_FLAGGED  # (N,)

        buried_mask = (
            active[:, None] & buried_eligible[None, :] & diff_mol & (d2 < bridge2)
        )

        # nbb: cross-mol, buried-eligible, within bb2
        bb2 = (4.0 * self.run.radmax + 4.0 * rp) ** 2
        nbb_mask = active[:, None] & buried_eligible[None, :] & diff_mol & (d2 < bb2)
        nbb = nbb_mask.sum(axis=1)  # (N,)

        # ── good_atom and access ──────────────────────────────────────────
        n_neighbors = neighbor_mask.sum(axis=1)  # (N,)
        atten6_no_nbb = (atten == ATTEN_6) & (nbb == 0)

        # access = 1 when active, not blocked by atten6_no_nbb, but no neighbors
        new_access = active & ~atten6_no_nbb & (n_neighbors == 0)
        atoms.access[:N] |= new_access.astype(np.int8)

        good_atom = active & ~atten6_no_nbb & (n_neighbors > 0)  # (N,)

        # ── build neighbor_array ──────────────────────────────────────────
        max_n = int(n_neighbors.max()) if n_neighbors.any() else 1
        neighbor_array = SimpleNeighborArray(N, max_n)

        n_buried = buried_mask.sum(axis=1)  # (N,)
        max_b = int(n_buried.max()) if n_buried.any() else 1
        buried_array = SimpleNeighborArray(N, max_b)

        for i in np.where(active)[0]:
            # neighbors sorted by ascending distance
            nidx = np.where(neighbor_mask[i])[0]
            if len(nidx):
                nidx = nidx[np.argsort(d2[i, nidx])]
                n = len(nidx)
                neighbor_array.xyz[i, :n] = xyz[nidx]
                neighbor_array.radius[i, :n] = radius[nidx]
                neighbor_array.natom[i, :n] = natom[nidx]
                neighbor_array.nneighbors[i] = n

            # buried (order doesn't matter)
            bidx = np.where(buried_mask[i])[0]
            if len(bidx):
                n = len(bidx)
                buried_array.xyz[i, :n] = xyz[bidx]
                buried_array.radius[i, :n] = radius[bidx]
                buried_array.natom[i, :n] = natom[bidx]
                buried_array.nneighbors[i] = n

        self.run.neighbor_array = neighbor_array
        self.run.buried_array = buried_array
        return good_atom

    def second_loop(self, atoms):
        """
        Vectorised second_loop over all atoms simultaneously.

        atoms : AtomArray — all good atoms to process (pre-filtered by good_atom mask)
        """
        if len(atoms) == 0:
            return

        rp = self.settings.rp
        na1 = atoms.natom  # (N,) original indices

        # ── neighbour data for all atoms at once ──────────────────────────
        nneigh = self.run.neighbor_array.nneighbors[na1]  # (N,)
        neigh_xyz = self.run.neighbor_array.xyz[na1]  # (N, K, 3)
        neigh_rad = self.run.neighbor_array.radius[na1]  # (N, K)
        neigh_natom = self.run.neighbor_array.natom[na1]  # (N, K)

        # ── forward-pair mask: only pairs where natom2 > natom1 ──────────
        fwd = neigh_natom > na1[:, None]  # (N, K)

        # ── geometry for all (atom, forward-neighbor) pairs ───────────────
        # Use np.where before sqrt to keep non-fwd entries finite (avoids NaN warnings).
        diff = neigh_xyz - atoms.xyz[:, None, :]  # (N, K, 3)
        dij_sq = np.einsum("nki,nki->nk", diff, diff)  # (N, K)
        dij = np.sqrt(np.where(fwd, dij_sq, 1.0))  # (N, K)

        eri = atoms.radius + rp  # (N,)
        erj = neigh_rad + rp  # (N, K) — NaN for padding

        asymm = np.where(fwd, (eri[:, None] ** 2 - erj**2) / dij, 0.0)  # (N, K)
        between = np.abs(asymm) < dij  # (N, K)
        tij = (atoms.xyz[:, None, :] + neigh_xyz) * 0.5 + (diff / dij[..., None]) * (
            asymm * 0.5
        )[..., None]  # (N, K, 3)

        far_sq = np.where(fwd, (eri[:, None] + erj) ** 2 - dij_sq, -1.0)  # (N, K)
        cont_sq = np.where(
            fwd, dij_sq - (atoms.radius[:, None] - neigh_rad) ** 2, -1.0
        )  # (N, K)

        geom_ok = fwd & (far_sq > 0.0) & (cont_sq > 0.0)  # (N, K)

        # ── single-neighbour shortcut ─────────────────────────────────────
        # Fires per-atom when nneigh <= 1 and there is at least one valid forward pair.
        shortcut = (nneigh <= 1) & geom_ok.any(axis=1)  # (N,)
        if shortcut.any():
            sc_na1 = na1[shortcut]
            self.run.atoms.access[sc_na1] = 1
            # First geom_ok forward neighbour for each shortcut atom
            first_k = np.argmax(geom_ok[shortcut], axis=1)  # (S,)
            first_na2 = neigh_natom[shortcut][np.arange(shortcut.sum()), first_k]
            self.run.atoms.access[first_na2.astype(np.int32)] = 1

        # ── flatten all non-shortcut, geom_ok pairs ───────────────────────
        pair_mask = geom_ok & ~shortcut[:, None]  # (N, K)
        i_idx, k_idx = np.where(pair_mask)

        if len(i_idx) == 0:
            return

        na1_f = na1[i_idx]  # (P,)
        na2_f = neigh_natom[i_idx, k_idx].astype(np.int32)  # (P,)
        dij_f = dij[i_idx, k_idx]  # (P,)
        uij_f = diff[i_idx, k_idx] / dij_f[:, None]  # (P, 3)
        tij_f = tij[i_idx, k_idx]  # (P, 3)
        rij_f = (
            0.5 * np.sqrt(far_sq[i_idx, k_idx]) * np.sqrt(cont_sq[i_idx, k_idx]) / dij_f
        )  # (P,)
        bet_f = between[i_idx, k_idx]  # (P,)

        # ── toroid queue ──────────────────────────────────────────────────
        a1_atten = atoms.atten[i_idx]  # (P,)
        a2_atten = self.run.atoms.atten[na2_f]  # (P,)
        need_tor = (a1_atten > ATTEN_BLOCKER) | (
            (a2_atten > ATTEN_BLOCKER) & (rp > 0.0)
        )
        for i in np.where(need_tor)[0]:
            self.run.toroid_queue.append(
                (
                    int(na1_f[i]),
                    int(na2_f[i]),
                    uij_f[i],
                    tij_f[i],
                    float(rij_f[i]),
                    bool(bet_f[i]),
                )
            )

        # ── vec_third_loop (single call across all pairs) ─────────────────
        access_natoms = self.vec_third_loop(
            self.run.atoms[na1_f],
            self.run.atoms[na2_f],
            uij_f,
            tij_f,
            rij_f,
        )
        if len(access_natoms):
            self.run.atoms.access[access_natoms] = 1

    def vec_third_loop(self, atom1s, atom2s, uij, tij, rij):
        """
        Vectorised third_loop over M (atom1, atom2) pairs simultaneously.

        atom1s : AtomArray  (M rows)
        atom2s : AtomArray  (M rows)
        uij    : (M, 3)  unit vector atom1→atom2
        tij    : (M, 3)  torus-circle centre
        rij    : (M,)    torus-circle radius

        Appends valid probes to self.run.probes.
        Returns a 1-D int32 array of natom indices that should gain access = 1.
        """
        M = len(atom1s)
        rp = self.settings.rp

        eri = atom1s.radius + rp  # (M,)
        erj = atom2s.radius + rp  # (M,)
        na1 = atom1s.natom  # (M,)  original atom indices
        na2 = atom2s.natom  # (M,)

        # ── neighbour data for every atom1 ───────────────────────────────
        neigh_xyz = self.run.neighbor_array.xyz[na1]  # (M, K, 3)
        neigh_rad = self.run.neighbor_array.radius[na1]  # (M, K)
        neigh_natom = self.run.neighbor_array.natom[na1]  # (M, K)
        K = neigh_xyz.shape[1]

        # ── initial (pair, atom3) validity mask ──────────────────────────
        # atom3 must be ordered after atom2 and be a real neighbour
        valid = (neigh_natom > na2[:, None]) & (neigh_natom >= 0)  # (M, K)

        erk = np.where(valid, neigh_rad + rp, np.inf)  # (M, K)

        diff_jk = neigh_xyz - atom2s.xyz[:, None, :]  # (M, K, 3)
        djk = np.linalg.norm(diff_jk, axis=-1)  # (M, K)
        valid &= djk < (erj[:, None] + erk)

        diff_ik = neigh_xyz - atom1s.xyz[:, None, :]  # (M, K, 3)
        dik = np.linalg.norm(diff_ik, axis=-1)  # (M, K)
        valid &= dik < (eri[:, None] + erk)

        # all-three-blocked filter
        safe_na3 = np.where(neigh_natom >= 0, neigh_natom, 0)
        a3_atten = self.run.atoms.atten[safe_na3]  # (M, K)
        all_blocked = (
            (atom1s.atten[:, None] <= ATTEN_BLOCKER)
            & (atom2s.atten[:, None] <= ATTEN_BLOCKER)
            & (a3_atten <= ATTEN_BLOCKER)
        )
        valid &= ~all_blocked

        if not np.any(valid):
            return np.empty(0, dtype=np.int32)

        # ── flatten to Q valid triples ────────────────────────────────────
        pair_idx, k_idx = np.where(valid)  # (Q,)
        Q = len(pair_idx)

        a1_xyz_q = atom1s.xyz[pair_idx]  # (Q, 3)
        a2_xyz_q = atom2s.xyz[pair_idx]  # (Q, 3)
        a3_xyz_q = neigh_xyz[pair_idx, k_idx]  # (Q, 3)
        a1_rad_q = atom1s.radius[pair_idx]  # (Q,)
        a3_rad_q = neigh_rad[pair_idx, k_idx]  # (Q,)
        na1_q = na1[pair_idx]  # (Q,)
        na2_q = na2[pair_idx]  # (Q,)
        na3_q = neigh_natom[pair_idx, k_idx]  # (Q,)
        eri_q = eri[pair_idx]  # (Q,)
        erk_q = a3_rad_q + rp  # (Q,)
        dik_q = dik[pair_idx, k_idx]  # (Q,)
        uij_q = uij[pair_idx]  # (Q, 3)
        tij_q = tij[pair_idx]  # (Q, 3)
        rij_q = rij[pair_idx]  # (Q,)

        # ── uik, dt, wijk, swijk ─────────────────────────────────────────
        uik_q = (a3_xyz_q - a1_xyz_q) / dik_q[:, None]  # (Q, 3)
        dt_q = np.einsum("qi,qi->q", uij_q, uik_q)  # (Q,)
        wijk_q = np.arccos(np.clip(dt_q, -1.0, 1.0))  # (Q,)
        swijk_q = np.sin(wijk_q)  # (Q,)

        degenerate = (dt_q >= 1.0) | (dt_q <= -1.0) | (wijk_q <= 0.0) | (swijk_q <= 0.0)

        # ── degenerate triples: check which ones kill their pair ──────────
        keep = ~degenerate  # (Q,)  start with non-degenerate only

        if np.any(degenerate):
            deg = degenerate
            dtijk2_deg = np.linalg.norm(tij_q[deg] - a3_xyz_q[deg], axis=-1)
            rkp2_deg = erk_q[deg] ** 2 - rij_q[deg] ** 2
            kills_q = np.zeros(Q, dtype=bool)
            kills_q[deg] = dtijk2_deg < rkp2_deg

            if np.any(kills_q):
                # Mark every triple that belongs to a killed pair
                kill_pair_m = np.zeros(M, dtype=bool)
                kill_pair_m[pair_idx[kills_q]] = True
                keep &= ~kill_pair_m[pair_idx]

        if not np.any(keep):
            return np.empty(0, dtype=np.int32)

        # ── apply keep filter ─────────────────────────────────────────────
        a1_xyz_q = a1_xyz_q[keep]
        a3_xyz_q = a3_xyz_q[keep]
        a1_rad_q = a1_rad_q[keep]
        a3_rad_q = a3_rad_q[keep]
        na1_q = na1_q[keep]
        na2_q = na2_q[keep]
        na3_q = na3_q[keep]
        eri_q = eri_q[keep]
        erk_q = erk_q[keep]
        dik_q = dik_q[keep]
        uij_q = uij_q[keep]
        tij_q = tij_q[keep]
        rij_q = rij_q[keep]
        uik_q = uik_q[keep]
        swijk_q = swijk_q[keep]
        pair_idx = pair_idx[keep]
        Q = len(pair_idx)

        # ── probe geometry ────────────────────────────────────────────────
        uijk_q = np.cross(uij_q, uik_q) / swijk_q[:, None]  # (Q, 3)
        utb_q = np.cross(uijk_q, uij_q)  # (Q, 3)

        asymm_q = (eri_q**2 - erk_q**2) / dik_q  # (Q,)
        tik_q = (a1_xyz_q + a3_xyz_q) * 0.5 + uik_q * (asymm_q * 0.5)[:, None]  # (Q, 3)

        # dt = uik · (tik - tij)  [the scalar third_loop computes this as
        # a component-wise product then sums, which equals the dot product]
        dt_b_q = np.einsum("qi,qi->q", uik_q, tik_q - tij_q)  # (Q,)
        bijk_q = tij_q + utb_q * (dt_b_q / swijk_q)[:, None]  # (Q, 3)

        hijk_sq_q = eri_q**2 - np.einsum(
            "qi,qi->q", bijk_q - a1_xyz_q, bijk_q - a1_xyz_q
        )  # (Q,)
        valid_h = hijk_sq_q > 0.0

        if not np.any(valid_h):
            return np.empty(0, dtype=np.int32)

        bijk_q = bijk_q[valid_h]
        uijk_q = uijk_q[valid_h]
        hijk_q = np.sqrt(hijk_sq_q[valid_h])
        na1_q = na1_q[valid_h]
        na2_q = na2_q[valid_h]
        na3_q = na3_q[valid_h]
        pair_idx = pair_idx[valid_h]
        Q = valid_h.sum()

        # ── two probe candidates per triple (isign = +1 and -1) ──────────
        # +1 half: a0=atom1, a1=atom2, a2=atom3
        # -1 half: a0=atom2, a1=atom1, a2=atom3
        bijk_2q = np.concatenate([bijk_q, bijk_q], axis=0)  # (2Q, 3)
        uijk_2q = np.concatenate([uijk_q, uijk_q], axis=0)  # (2Q, 3)
        hijk_2q = np.concatenate([hijk_q, hijk_q])  # (2Q,)
        na1_2q = np.concatenate([na1_q, na1_q])  # (2Q,)
        na2_2q = np.concatenate([na2_q, na2_q])  # (2Q,)
        na3_2q = np.concatenate([na3_q, na3_q])  # (2Q,)
        isign_2q = np.concatenate([np.ones(Q), -np.ones(Q)])  # (2Q,)

        pijk_2q = bijk_2q + uijk_2q * (hijk_2q * isign_2q)[:, None]  # (2Q, 3)
        alt_2q = uijk_2q * isign_2q[:, None]  # (2Q, 3)

        probe_a0 = np.where(isign_2q > 0, na1_2q, na2_2q)  # (2Q,)
        probe_a1 = np.where(isign_2q > 0, na2_2q, na1_2q)  # (2Q,)
        probe_a2 = na3_2q  # (2Q,)

        # ── collision check ───────────────────────────────────────────────
        # na1_2q indexes the atom whose neighbour list we check against
        coll_xyz = self.run.neighbor_array.xyz[na1_2q]  # (2Q, K, 3)
        coll_rad = self.run.neighbor_array.radius[na1_2q]  # (2Q, K)
        coll_natom = self.run.neighbor_array.natom[na1_2q]  # (2Q, K)

        # Exclude the two generating atoms (atom2 and atom3) and NaN padding
        # slots (natom < 0).  Active slots are guaranteed finite xyz and radius.
        exclude = (
            (coll_natom == na2_2q[:, None])
            | (coll_natom == na3_2q[:, None])
            | (coll_natom < 0)
        )

        # Compute distances only for active (non-excluded) slots, avoiding all
        # NaN arithmetic on padding entries.
        rows, cols = np.where(~exclude)  # (S,)
        diff_active = pijk_2q[rows] - coll_xyz[rows, cols]  # (S, 3)
        d2_active = np.einsum(
            "ij,ij->i", diff_active, diff_active
        )  # (S,) — single pass, no intermediate
        within_active = d2_active <= (coll_rad[rows, cols] + rp) ** 2

        # Scatter: any hit on a probe row marks that probe as colliding.
        collision = np.zeros(len(pijk_2q), dtype=bool)
        collision[rows[within_active]] = True

        valid_probe = ~collision
        if not np.any(valid_probe):
            return np.empty(0, dtype=np.int32)

        # ── write probes ──────────────────────────────────────────────────
        pijk_f = pijk_2q[valid_probe]
        alt_f = alt_2q[valid_probe]
        hijk_f = hijk_2q[valid_probe]
        probe_a0_f = probe_a0[valid_probe].astype(np.int32)
        probe_a1_f = probe_a1[valid_probe].astype(np.int32)
        probe_a2_f = probe_a2[valid_probe].astype(np.int32)

        self.run.probes.extend_from_arrays(
            probe_a0_f,
            probe_a1_f,
            probe_a2_f,
            hijk_f,
            pijk_f,
            alt_f,
        )

        # ── return natoms that gained access ─────────────────────────────
        return np.unique(np.concatenate([probe_a0_f, probe_a1_f, probe_a2_f]))

    def generate_convex_surface(self, atoms):
        N = atoms.xyz.shape[0]

        north = np.zeros((N, 3))
        north[:] = np.array([0.0, 0.0, 1.0])

        south = np.zeros((N, 3))
        south[:] = np.array([0.0, 0.0, -1.0])

        eqvec = np.zeros((N, 3))
        eqvec[:] = np.array([1.0, 0.0, 0.0])

        ri = atoms.radius  # (N,)
        eri = atoms.radius + self.settings.rp

        nneigh = self.run.neighbor_array.nneighbors[atoms.natom]  # (N,)
        has_neigh = nneigh > 0

        if np.any(has_neigh):
            neigh_xyz = self.run.neighbor_array.xyz[atoms.natom][:, 0]  # (N,3)
            neigh_rad = self.run.neighbor_array.radius[atoms.natom][:, 0]  # (N,)

            # direction to neighbor
            north_vec = atoms.xyz - neigh_xyz
            north_vec /= np.linalg.norm(north_vec, axis=-1, keepdims=True)

            north[has_neigh] = north_vec[has_neigh]

            vtemp = np.stack(
                [
                    north[:, 1] ** 2 + north[:, 2] ** 2,
                    north[:, 0] ** 2 + north[:, 2] ** 2,
                    north[:, 0] ** 2 + north[:, 1] ** 2,
                ],
                axis=-1,
            )

            vtemp /= np.linalg.norm(vtemp, axis=-1, keepdims=True)

            dt = np.sum(vtemp * north, axis=-1)

            replace = np.abs(dt) > 0.99
            vtemp[replace] = np.array([1.0, 0.0, 0.0])

            eq = np.cross(north, vtemp)
            eq /= np.linalg.norm(eq, axis=-1, keepdims=True)

            eqvec[has_neigh] = eq[has_neigh]

            vql = np.cross(eqvec, north)

            rj = neigh_rad
            erj = neigh_rad + self.settings.rp

            dij = np.linalg.norm(neigh_xyz - atoms.xyz, axis=-1)
            uij = (neigh_xyz - atoms.xyz) / dij[:, None]

            asymm = (eri * eri - erj * erj) / dij
            tij = ((atoms.xyz + neigh_xyz) * 0.5) + uij * (asymm[:, None] * 0.5)

            far_sq = (eri + erj) ** 2 - dij * dij
            if np.any(far_sq <= 0):
                raise RuntimeError("Imaginary _far_")

            far = np.sqrt(far_sq)

            contain_sq = dij * dij - (ri - rj) ** 2
            if np.any(contain_sq <= 0):
                raise RuntimeError("Imaginary contain")

            contain = np.sqrt(contain_sq)

            rij = 0.5 * far * contain / dij

            pij = tij + vql * rij[:, None]
            south_vec = (pij - atoms.xyz) / eri[:, None]

            south[has_neigh] = south_vec[has_neigh]

        # ---------------------------------------------------------
        # Latitude arcs for ALL atoms simultaneously
        # ---------------------------------------------------------

        o = np.zeros((N, 3))

        cs, lats = self.vec_sub_arc(o, ri, eqvec, atoms.density, north, south)

        # lats: (N, K, 3)  K <= MAX_SUBDIV
        valid_lats = ~np.isnan(lats[..., 0])

        # dt per atom per latitude
        dt = np.sum(lats * north[:, None, :], axis=-1)  # (N, K)

        cen = atoms.xyz[:, None, :] + dt[..., None] * north[:, None, :]

        rad_sq = ri[:, None] ** 2 - dt * dt
        valid_rad = rad_sq > 0

        rad = np.zeros_like(rad_sq)
        rad[valid_rad] = np.sqrt(rad_sq[valid_rad])

        # ---------------------------------------------------------
        # Generate ALL circle points, skipping zero-radius rows
        # ---------------------------------------------------------

        M = lats.shape[1]
        flat_rad = rad.reshape(-1)
        flat_cen = cen.reshape(-1, 3)
        flat_north = np.repeat(north, M, axis=0)
        flat_density = np.repeat(atoms.density, M)
        active = flat_rad > 0

        if active.any():
            ps_a, pts_a = self.vec_sub_cir(
                flat_cen[active],
                flat_rad[active],
                flat_north[active],
                flat_density[active],
            )
            K = pts_a.shape[1]
            full_pts = np.full((N * M, K, 3), np.nan)
            full_pts[active] = pts_a
            full_ps = np.zeros(N * M)
            full_ps[active] = ps_a
        else:
            K = 0
            full_pts = np.empty((N * M, 0, 3))
            full_ps = np.zeros(N * M)

        points = full_pts.reshape(N, M, K, 3)
        ps = full_ps.reshape(N, M)

        valid_points = ~np.isnan(points[..., 0])

        area = ps * cs[:, None]

        # ---------------------------------------------------------
        # Project outward
        # ---------------------------------------------------------

        points_flat = points[valid_points]
        atom_idx, lat_idx, circ_idx = np.where(valid_points)

        pcen = (
            atoms.xyz[atom_idx]
            + (points_flat - atoms.xyz[atom_idx])
            * (eri[atom_idx] / ri[atom_idx])[:, None]
        )

        # ---------------------------------------------------------
        # Collision check batched
        # ---------------------------------------------------------

        collisions = self.vec_check_point_collision(
            pcen[..., None, :],
            self.run.neighbor_array.xyz[atoms.natom][atom_idx],
            self.run.neighbor_array.radius[atoms.natom][atom_idx],
        )

        keep = ~collisions

        points_keep = points_flat[keep]
        pcen_keep = pcen[keep]
        atom_keep = atom_idx[keep]
        lat_keep = lat_idx[keep]

        if points_keep.shape[0] == 0:
            return 1

        self.run.results.dots.convex += points_keep.shape[0]

        self.add_dots(
            atoms.molecule[atom_keep],
            1,
            points_keep,
            area[atom_keep, lat_keep],
            pcen_keep,
            atoms.natom[atom_keep],
        )

        return 1

    def vec_check_point_collision(self, pcen, xyzs, rads):
        # skip first neighbor (matches C++ begin()+1)
        dists2 = np.square(xyzs - pcen).sum(axis=-1)
        collision = (dists2 <= (rads + self.settings.rp) ** 2) & ~np.isnan(rads)

        return collision[..., 1:].any(axis=-1)

    import math

    def generate_toroidal_surface(
        self,
        atom1,
        atom2,
        uij,  # (N,3)
        tij,  # (N,3)
        rij,  # (N,)
        between,  # (N,)
    ):
        N = atom1.xyz.shape[0]

        neigh_xyz = self.run.neighbor_array.xyz[atom1.natom]  # (N,K,3)
        neigh_rad = self.run.neighbor_array.radius[atom1.natom]  # (N,K)
        neigh_natom = self.run.neighbor_array.natom[atom1.natom]  # (N,K)

        density = (atom1.density + atom2.density) * 0.5

        eri = atom1.radius + self.settings.rp
        erj = atom2.radius + self.settings.rp

        rci = rij * atom1.radius / eri
        rcj = rij * atom2.radius / erj
        rb = np.maximum(rij - self.settings.rp, 0.0)

        rs = (rci + 2 * rb + rcj) * 0.25
        e = rs / rij
        edens = e * e * density

        # ---------------------------------------------------------
        # Subdivide torus circle (batched)
        # ---------------------------------------------------------

        ts, subs = self.vec_sub_cir(tij, rij, uij, edens)

        # subs: (N, M, 3)
        valid_sub = ~np.isnan(subs[..., 0])

        if not np.any(valid_sub):
            return (np.zeros(N, dtype=bool), np.zeros(N, dtype=bool))

        # Flatten valid subdivisions
        atom_idx, sub_idx = np.where(valid_sub)
        pij = subs[atom_idx, sub_idx]  # (Q,3)

        # ---------------------------------------------------------
        # Neighbor collision test (vectorized)
        # ---------------------------------------------------------

        neigh_xyz_flat = neigh_xyz[atom_idx]  # (Q,K,3)
        neigh_rad_flat = neigh_rad[atom_idx]
        neigh_natom_flat = neigh_natom[atom_idx]

        # Exclude padding slots (natom < 0) and atom2 itself, then compute
        # distances only on real neighbor entries — no NaN arithmetic.
        exclude = (neigh_natom_flat < 0) | (
            neigh_natom_flat == atom2.natom[atom_idx, None]
        )
        rows, cols = np.where(~exclude)  # (S,)
        diff_active = pij[rows] - neigh_xyz_flat[rows, cols]  # (S, 3)
        d2_active = np.einsum(
            "ij,ij->i", diff_active, diff_active
        )  # (S,) — single pass
        within_active = d2_active < (neigh_rad_flat[rows, cols] + self.settings.rp) ** 2

        too_close_any = np.zeros(len(pij), dtype=bool)
        too_close_any[rows[within_active]] = True

        valid_point = ~too_close_any

        valid_point &= ~(
            (atom1[atom_idx].atten == ATTEN_6)
            & (atom2[atom_idx].atten == ATTEN_6)
            & (self.run.buried_array.nneighbors[atom1[atom_idx].natom] == 0)
        )

        if not np.any(valid_point):
            return (np.zeros(N, dtype=bool), np.zeros(N, dtype=bool))

        pij = pij[valid_point]
        atom_idx = atom_idx[valid_point]

        # Mark access
        atom1_has_access = np.zeros(N, dtype=bool)
        atom2_has_access = np.zeros(N, dtype=bool)

        atom1_has_access[atom_idx] = True
        atom2_has_access[atom_idx] = True

        # ---------------------------------------------------------
        # Geometry
        # ---------------------------------------------------------

        pi = (atom1.xyz[atom_idx] - pij) / eri[atom_idx, None]
        pj = (atom2.xyz[atom_idx] - pij) / erj[atom_idx, None]

        axis = np.cross(pi, pj)
        axis /= np.linalg.norm(axis, axis=-1, keepdims=True)

        dtq = self.settings.rp**2 - rij[atom_idx] ** 2
        pcusp = (dtq > 0) & between[atom_idx]

        pqi = np.empty_like(pi)
        pqj = np.empty_like(pi)

        if np.any(pcusp):
            dtq_s = np.sqrt(dtq[pcusp])
            qij = tij[atom_idx[pcusp]] - uij[atom_idx[pcusp]] * dtq_s[:, None]
            qjk = tij[atom_idx[pcusp]] + uij[atom_idx[pcusp]] * dtq_s[:, None]

            pqi[pcusp] = (qij - pij[pcusp]) / self.settings.rp
            pqj[pcusp] = 0.0

        if np.any(~pcusp):
            tmp = pi[~pcusp] + pj[~pcusp]
            tmp /= np.linalg.norm(tmp, axis=-1, keepdims=True)
            pqi[~pcusp] = tmp
            pqj[~pcusp] = tmp

        # Reject invalid dot cases
        dt1 = np.sum(pqi * pi, axis=-1)
        dt2 = np.sum(pqj * pj, axis=-1)

        valid_angle = (np.abs(dt1) < 1.0) & (np.abs(dt2) < 1.0)

        if not np.any(valid_angle):
            return atom1_has_access, atom2_has_access

        pij = pij[valid_angle]
        axis = axis[valid_angle]
        pi = pi[valid_angle]
        pj = pj[valid_angle]
        pqi = pqi[valid_angle]
        pqj = pqj[valid_angle]
        atom_idx = atom_idx[valid_angle]

        # ---------------------------------------------------------
        # Arc generation (batched)
        # ---------------------------------------------------------

        areas_total = 0

        # ---- atom1 arc ----
        mask1 = atom1.atten[atom_idx] >= ATTEN_2
        if np.any(mask1):
            ps, points = self.vec_sub_arc(
                pij[mask1],
                np.full(np.sum(mask1), self.settings.rp),
                axis[mask1],
                density[atom_idx[mask1]],
                pi[mask1],
                pqi[mask1],
            )

            dist = self.vec_distance_point_to_line(
                tij[atom_idx[mask1]][..., None, :],
                uij[atom_idx[mask1]][..., None, :],
                points,
            )

            expanded_pij = np.zeros(points.shape)
            expanded_pij[:] = pij[mask1][:, None, :]
            areas = (
                ps[:, None]
                * ts[atom_idx[mask1], None]
                * dist
                / rij[atom_idx[mask1], None]
            )
            valid = ~np.isnan(areas)

            K = int(valid.sum())
            self.run.results.dots.toroidal += K
            if K > 0:
                mol = np.broadcast_to(
                    atom1.molecule[atom_idx][mask1][:, None], valid.shape
                )[valid]
                nat = np.broadcast_to(
                    atom1.natom[atom_idx][mask1][:, None], valid.shape
                )[valid]
                self.add_dots(
                    mol, 2, points[valid], areas[valid], expanded_pij[valid], nat
                )

            # areas_total += np.sum(valid)

        # ---- atom2 arc ----
        mask2 = atom2.atten[atom_idx] >= ATTEN_2
        if np.any(mask2):
            ps, points = self.vec_sub_arc(
                pij[mask2],
                np.full(np.sum(mask2), self.settings.rp),
                axis[mask2],
                density[atom_idx[mask2]],
                pqj[mask2],
                pj[mask2],
            )

            dist = self.vec_distance_point_to_line(
                tij[atom_idx[mask2]][..., None, :],
                uij[atom_idx[mask2]][..., None, :],
                points,
            )

            expanded_pij = np.zeros(points.shape)
            expanded_pij[:] = pij[mask2][:, None, :]
            areas = (
                ps[:, None]
                * ts[atom_idx[mask2], None]
                * dist
                / rij[atom_idx[mask2], None]
            )
            valid = ~np.isnan(areas)

            K = int(valid.sum())
            self.run.results.dots.toroidal += K
            if K > 0:
                mol = np.broadcast_to(
                    atom1.molecule[atom_idx][mask2][:, None], valid.shape
                )[valid]
                nat = np.broadcast_to(
                    atom2.natom[atom_idx][mask2][:, None], valid.shape
                )[valid]
                self.add_dots(
                    mol, 2, points[valid], areas[valid], expanded_pij[valid], nat
                )
            # areas_total += np.sum(valid)

        # self.run.results.dots.toroidal += areas_total

        return atom1_has_access, atom2_has_access

    def generate_concave_surface(self):
        probes = self.run.probes
        if not probes:
            return 1

        rp = self.settings.rp
        rp2 = rp * rp

        # ---------------------------------------------------------
        # Pull probe data into arrays
        # ---------------------------------------------------------

        P = len(probes)

        pijk = (
            probes.point_xyz
        )  # np.array([p.point.to_numpy() for p in probes])        # (P,3)
        uijk = (
            probes.alt_xyz
        )  # np.array([p.alt.to_numpy() for p in probes])          # (P,3)
        hijk = (
            probes.height
        )  # np.array([p.height for p in probes])                  # (P,)

        atom_natom = [probes.atom_idx_0, probes.atom_idx_1, probes.atom_idx_2]
        atom_arrays = [self.run.atoms[natom] for natom in atom_natom]

        atom_xyz = np.stack([arr.xyz for arr in atom_arrays], axis=1)
        atom_radius = np.stack([arr.radius for arr in atom_arrays], axis=-1)
        atom_density = np.stack([arr.density for arr in atom_arrays], axis=-1)
        atom_atten = np.stack([arr.atten for arr in atom_arrays], axis=-1)
        atom_molecule = np.stack([arr.molecule for arr in atom_arrays], axis=-1)

        density = atom_density.mean(axis=1)  # (P,)

        # ---------------------------------------------------------
        # Identify low probes
        # ---------------------------------------------------------

        low_mask = hijk < rp
        low_points = pijk[low_mask]

        # ---------------------------------------------------------
        # Skip fully attenuated probes
        # ---------------------------------------------------------

        skip = np.all(atom_atten == ATTEN_6, axis=1)
        active = ~skip

        if not np.any(active):
            return 1

        # ---------------------------------------------------------
        # Nearby low probes (vectorized distance)
        # ---------------------------------------------------------

        nears_mask = None
        if np.any(low_mask):
            d2 = np.sum(
                (pijk[:, None, :] - low_points[None, :, :]) ** 2, axis=-1
            )  # (P, L)

            # nears_mask = d2 <= 4 * rp2

            low_indices = np.where(low_mask)[0]  # (L,) original probe indices
            nears_mask = d2 <= 4 * rp2
            nears_mask[low_indices, np.arange(len(low_indices))] = False  # exclude self

        # ---------------------------------------------------------
        # Vectors from probe to atoms
        # ---------------------------------------------------------

        vp = atom_xyz - pijk[:, None, :]
        vp /= np.linalg.norm(vp, axis=-1, keepdims=True)

        vectors = np.stack(
            [
                np.cross(vp[:, 0], vp[:, 1]),
                np.cross(vp[:, 1], vp[:, 2]),
                np.cross(vp[:, 2], vp[:, 0]),
            ],
            axis=1,
        )

        vectors /= np.linalg.norm(vectors, axis=-1, keepdims=True)

        # ---------------------------------------------------------
        # Highest vertex
        # ---------------------------------------------------------

        dt = np.sum(uijk[:, None, :] * vp, axis=-1)  # (P,3)
        mm = np.argmax(dt, axis=1)

        south = -uijk

        axis = np.cross(vp[np.arange(P), mm], south)
        axis /= np.linalg.norm(axis, axis=-1, keepdims=True)

        # ---------------------------------------------------------
        # Latitude arcs (batched)
        # ---------------------------------------------------------

        o = np.zeros_like(pijk)

        cs, lats = self.vec_sub_arc(
            o, np.full(P, rp), axis, density, vp[np.arange(P), mm], south
        )

        valid_lat = ~np.isnan(lats[..., 0])

        # ---------------------------------------------------------
        # Circle subdivisions
        # ---------------------------------------------------------

        dt_lat = np.sum(lats * south[:, None, :], axis=-1)
        cen = south[:, None, :] * dt_lat[..., None]

        rad_sq = rp2 - dt_lat**2
        valid_rad = rad_sq > 0
        rad = np.sqrt(np.clip(rad_sq, 0, None))

        M = lats.shape[1]
        flat_rad = rad.reshape(-1)
        flat_cen = cen.reshape(-1, 3)
        flat_south = np.repeat(south, M, axis=0)
        flat_density_r = np.repeat(density, M)
        active = flat_rad > 0

        if active.any():
            ps_a, pts_a = self.vec_sub_cir(
                flat_cen[active],
                flat_rad[active],
                flat_south[active],
                flat_density_r[active],
            )
            K = pts_a.shape[1]
            full_pts = np.full((P * M, K, 3), np.nan)
            full_pts[active] = pts_a
            full_ps = np.zeros(P * M)
            full_ps[active] = ps_a
        else:
            K = 0
            full_pts = np.empty((P * M, 0, 3))
            full_ps = np.zeros(P * M)

        points = full_pts.reshape(P, M, K, 3)
        ps = full_ps.reshape(P, M)

        valid_points = ~np.isnan(points[..., 0])

        area = ps * cs[:, None]

        # ---------------------------------------------------------
        # Flatten all valid geometry
        # ---------------------------------------------------------

        idx_probe, idx_lat, idx_pt = np.where(valid_points)
        pts = points[idx_probe, idx_lat, idx_pt]
        cen_flat = pijk[idx_probe]

        # Vector rejection test
        vecs = vectors[idx_probe]
        bail = np.any(np.sum(pts[:, None, :] * vecs, axis=-1) >= 0, axis=1)

        pts = pts[~bail]
        idx_probe = idx_probe[~bail]
        idx_lat = idx_lat[~bail]

        pts = pts + cen_flat[~bail]

        # ---------------------------------------------------------
        # Low-probe collision
        # ---------------------------------------------------------

        if nears_mask is not None:
            near_any = nears_mask[idx_probe].any(axis=1)

            coll = self.check_probe_collision_vectorized(
                pts, low_points, nears_mask[idx_probe], rp2
            )

            reject = (hijk[idx_probe] < rp) & near_any & coll

            pts = pts[~reject]
            idx_probe = idx_probe[~reject]
            idx_lat = idx_lat[~reject]

        # ---------------------------------------------------------
        # Closest atom selection
        # ---------------------------------------------------------

        d = (
            np.linalg.norm(pts[:, None, :] - atom_xyz[idx_probe], axis=-1)
            - atom_radius[idx_probe]
        )

        mc = np.argmin(d, axis=1)

        # ---------------------------------------------------------
        # Final dot creation
        # ---------------------------------------------------------

        self.run.results.dots.concave += pts.shape[0]

        atom_natom_stack = np.stack(atom_natom, axis=1)  # (P, 3)
        self.add_dots(
            atom_molecule[idx_probe, mc],
            3,
            pts,
            area[idx_probe, idx_lat],
            pijk[idx_probe],
            atom_natom_stack[idx_probe, mc],
        )

        return 1

    def check_probe_collision_vectorized(
        self,
        points,  # (N,3)
        near_points,  # (M,3)
        near_mask,  # (N,M) bool
        r2,  # scalar
    ):
        """
        Returns (N,) boolean array.
        True if point i collides with ANY allowed near_point.
        """

        N = points.shape[0]

        # Find only valid (i,j) pairs
        rows, cols = np.where(near_mask)

        if rows.size == 0:
            return np.zeros(N, dtype=bool)

        # Gather only those coordinates
        p_sel = points[rows]  # (K,3)
        q_sel = near_points[cols]  # (K,3)

        # Compute squared distances only for valid pairs
        diff = p_sel - q_sel
        d2 = np.einsum("ij,ij->i", diff, diff)  # fast rowwise dot

        # Determine which pairs collide
        colliding_pairs = d2 < r2

        if not np.any(colliding_pairs):
            return np.zeros(N, dtype=bool)

        # Mark which points had at least one collision
        collided_points = np.zeros(N, dtype=bool)
        collided_points[rows[colliding_pairs]] = True

        return collided_points

    def add_dots(self, molecule, type_, coor, area, pcen, atom_indices):
        """
        Vectorised batch version of add_dot.

        molecule     : (N,) int   — 0 or 1
        type_        : int scalar — 1=convex, 2=toroidal, 3=concave
        coor         : (N, 3)    — surface dot coordinates
        area         : (N,)      — area per dot
        pcen         : (N, 3)    — probe / atom centre (for normal and burial)
        atom_indices : (N,) int  — index into self.run.atoms (== natom == _idx)
        """
        N = len(coor)
        if N == 0:
            return

        pradius = self.settings.rp
        atom_indices = np.asarray(atom_indices, dtype=np.int32)
        molecule = np.asarray(molecule, dtype=np.int8)

        # ── outward normal ────────────────────────────────────────────────
        if pradius <= 0:
            outnml = coor - self.run.atoms.xyz[atom_indices]  # (N, 3)
        else:
            outnml = (pcen - coor) / pradius  # (N, 3)

        # ── buried determination ──────────────────────────────────────────
        bxyz = self.run.buried_array.xyz[atom_indices]  # (N, max_b, 3)
        brad = self.run.buried_array.radius[atom_indices]  # (N, max_b)
        d2 = np.square(bxyz - pcen[:, None, :]).sum(axis=-1)  # (N, max_b)
        buried = (d2 <= (brad + pradius) ** 2).any(axis=-1).astype(np.int8)  # (N,)

        # ── partition by molecule and batch-append ────────────────────────
        type_arr = np.full(N, type_, dtype=np.int8)
        for mol in (0, 1):
            mask = molecule == mol
            if not mask.any():
                continue
            self.run.dots[mol].extend(
                coor[mask],
                outnml[mask],
                area[mask],
                buried[mask],
                type_arr[mask],
                atom_indices[mask],
            )

    def vec_distance_point_to_line(self, cen, axis, pnt):
        vec = pnt - cen
        dt = (vec * axis).sum(axis=-1)  # vec.dot(axis)
        d2 = np.square(vec).sum(axis=-1) - dt * dt

        return np.where(d2 < 0.0, 0, np.sqrt(d2))
        # if d2 < 0.0:
        #     return 0.0

        # return math.sqrt(d2)

    def distance_point_to_line(self, cen, axis, pnt):
        vec = pnt - cen
        dt = vec.dot(axis)
        d2 = vec.magnitude_squared() - dt * dt

        if d2 < 0.0:
            return 0.0

        return math.sqrt(d2)

    import numpy as np

    def vec_sub_arc(self, cen, rad, axis, density, x, v):
        """
        cen, axis, x, v: (..., 3)
        rad, density: (...)
        Returns:
            ps: (...)
            points: (..., K, 3)  where K <= MAX_SUBDIV
        """

        # y = axis × x
        y = np.cross(axis, x)

        dt1 = np.sum(v * x, axis=-1)
        dt2 = np.sum(v * y, axis=-1)

        angle = np.arctan2(dt2, dt1)
        angle = np.where(angle < 0.0, angle + 2 * np.pi, angle)

        angle = np.where(np.isclose(dt1, 0) & np.isclose(dt2, 0), np.nan, angle)

        return self.vec_sub_div(cen, rad, x, y, angle, density)

    def vec_sub_div(self, cen, rad, x, y, angle, density):
        """
        cen, x, y: (..., 3)
        rad, angle, density: (...)

        Returns:
            ps: (...)
            points: (..., K, 3)  where K = min(MAX_SUBDIV, max subdivisions needed)
        """

        rad = np.asarray(rad)
        density = np.asarray(density)
        angle = np.asarray(angle)

        base_shape = rad.shape

        # Angular spacing — guard div-by-zero; invalid elements produce inf/nan
        with np.errstate(divide="ignore", invalid="ignore"):
            delta = 1.0 / (np.sqrt(density) * rad)
            raw = angle / delta  # exact subdivisions needed per element

        # Trim the inner dimension to only as many slots as the worst-case element
        # needs, rather than always allocating MAX_SUBDIV=100.
        valid_raw = np.isfinite(raw) & (raw > 0)
        if not valid_raw.any():
            return np.zeros(base_shape), np.full(base_shape + (0, 3), np.nan)

        max_count = int(min(MAX_SUBDIV, math.ceil(float(raw[valid_raw].max()))))
        if max_count == 0:
            return np.zeros(base_shape), np.full(base_shape + (0, 3), np.nan)

        i = np.arange(max_count)

        # a_i = delta*(i + 1/2)
        a = delta[..., None] * (i + 0.5)

        # Mask where subdivision exceeds angle
        mask = a <= angle[..., None]

        # cos/sin
        c = rad[..., None] * np.cos(a)
        s = rad[..., None] * np.sin(a)

        # Expand vectors
        cen_exp = cen[..., None, :]
        x_exp = x[..., None, :]
        y_exp = y[..., None, :]

        points = cen_exp + x_exp * c[..., None] + y_exp * s[..., None]

        # Apply mask → nan where invalid
        points = np.where(mask[..., None], points, np.nan)

        # Count valid points per element; ps = arc_length / count
        counts = np.sum(mask, axis=-1)
        ps = np.where(counts > 0, rad * angle / np.clip(counts, 0.01, None), 0.0)

        return ps, points

    def vec_sub_cir(self, cen, rad, axis, density):
        """
        cen, axis: (..., 3)
        rad, density: (...)

        Returns:
            ps: (...)
            points: (..., MAX_SUBDIV, 3)
        """

        axis = axis / np.linalg.norm(axis, axis=-1, keepdims=True)

        # Build v1
        v1 = np.stack(
            [
                axis[..., 1] ** 2 + axis[..., 2] ** 2,
                axis[..., 0] ** 2 + axis[..., 2] ** 2,
                axis[..., 0] ** 2 + axis[..., 1] ** 2,
            ],
            axis=-1,
        )

        v1 = v1 / np.linalg.norm(v1, axis=-1, keepdims=True)

        dt = np.sum(v1 * axis, axis=-1)

        # Replace near-parallel cases
        replace = np.abs(dt) > 0.99
        v1 = np.where(replace[..., None], np.array([1.0, 0.0, 0.0]), v1)

        v2 = np.cross(axis, v1)
        v2 = v2 / np.linalg.norm(v2, axis=-1, keepdims=True)

        x = np.cross(axis, v2)
        x = x / np.linalg.norm(x, axis=-1, keepdims=True)

        y = np.cross(axis, x)

        angle = np.full_like(rad, 2 * np.pi)

        return self.vec_sub_div(cen, rad, x, y, angle, density)


# if __name__ == '__main__':

#     import sys
#     from pyrosetta import init, pose_from_file
#     init('-mute all')

#     pdb = sys.argv[1]

#     pose = pose_from_file(pdb)

#     binder_xyz, binder_radii, target_xyz, target_radii= partition_pose(pose)
#     cms, per_target_atom_cms, calc = calculate_contact_ms(binder_xyz, binder_radii, target_xyz, target_radii)

#     print(cms)


#     d0    = calc.run.dots[0]
#     dots0 = d0.coor_xyz

#     d1    = calc.run.dots[1]
#     dots1 = d1.coor_xyz
