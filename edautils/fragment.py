from typing import Dict
import numpy as np
from rdkit import Chem
from openbabel import openbabel


def _is_sp3_carbon(atom):
    if atom.GetSymbol() != 'C':
        return False
    elif len(atom.GetNeighbors()) == 4:
        return True
    elif atom.GetHybridization() == Chem.HybridizationType.SP3:
        return True
    else:
        return False


def find_rings(
    mol: Chem.Mol, 
    force_exclude_macro: bool = True,
    force_include_aromatic: bool = True,
    force_include_small_ring: bool = True,
    macro_thresh: int = 8, 
    small_ring_thresh: int = 4,
    merge_rings: bool = True, 
    merge_thresh: int = 1, 
    exclude_non_hetero_ring: bool = True,
    sp3c_thresh: int = 2
):
    """
    Find rings that are relevant 'electronically' in a given molecule

    Parameters
    ----------
    max_ring_size: int
        Rings larger than `max_ring_size` will not be considered as a ring, to
        avoid including macrocycles. Default is 8.
    force_exclude_macro: bool
        Forcibly exclude macrocycles. A ring is considered to be macrocycle if it 
        contains more than `macro_thresh` atoms. Default is True.
    force_include_aromatic: bool
        Forcibly include aromatic rings. A ring is considered to be aromatic if
        all atoms in that ring are aromatic. Default is True.
    force_include_small_ring: bool
        Forcibly include small-ring system. A ring is considered to be small if 
        if contains equal or less than `small_ring_thresh` atoms. Default is True.
    
    small_ring_thresh: int
        If a ring contains less than `min_ring_size` atoms, it will be considered
        as a ring anyhow.
    merge_rings: bool
        Whether to merge rings that share more than `merge_thresh` atoms.
        Default is True.
    merge_rings_thresh: int
        If set to 0, spiral/bridged/fused rings will be merged.
        If set to 1, bridged/fused rings will be merged, but not spirals.
        Default is 1.
    exclude_non_hetero_ring: bool
        For rings that contains equal or more than `min_ring_size` atoms and equal or less
        than `max_ring_size` atoms, whether to exclue it if it contains more than
        `sp3c_thresh` sp3 carbon atoms. Default is True.
    sp3c_thresh: int
        If a ring contains more than `sp3c_thresh` sp3 carbon atoms, the ring will be flagged
        as "non-hetero" and will be excluded if `exclude_non_hetero_ring` is True.
    """
    rings = []
    for r in Chem.GetSymmSSSR(mol):
        if force_exclude_macro and len(r) > macro_thresh: # never record macrocycles
            continue
        if force_include_aromatic and all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in r):
            rings.append(set(r))
            continue
        if force_include_small_ring and len(r) <= small_ring_thresh:
            rings.append(set(r))
            continue
        if exclude_non_hetero_ring and len([idx for idx in r if _is_sp3_carbon(mol.GetAtomWithIdx(idx))]) > sp3c_thresh:
            continue
        rings.append(set(r))

    if merge_rings:
        flag = True  
        while flag:
            flag = False
            for i in range(len(rings)):
                if len(rings[i]) == 0: continue
                for j in range(i + 1, len(rings)):
                    shared_atoms = rings[i] & rings[j]
                    if len(shared_atoms) > merge_thresh:
                        rings[i].update(rings[j])
                        rings[j] = set()
                        flag = True
    rings = [set(r) for r in rings if len(r) > 0]
    return rings


FG_SMARTS = {
    'HETEROATOM': '[!#6;!#1]',
    'DOUBLE_TRIPLE_BOND': '[*]=,#[*]',
    'ACETAL': '[CX4]([O,N,S])[O,N,S]',
    'HALO': '[CX4]([F,Cl,Br,I])([F,Cl,Br,I])'
}
    
def find_fgs(mol: Chem.Mol, custom_fgs: Dict[str, str] = dict()):

    if custom_fgs:
        patts = [Chem.MolFromSmarts(v) for v in custom_fgs.values()]
    else:
        patts = [Chem.MolFromSmarts(v) for v in FG_SMARTS.values()]


    # identify functional atoms and merge connected ones
    fgs = []

    marks = set()
    for patt in patts:  # mark functional atoms
        for sub in mol.GetSubstructMatches(patt):
            marks.update(sub)
    
    atom2fg = [None for _ in range(mol.GetNumAtoms())]  # atom2fg[i]: list of i-th atom's FG idx
    for atom in marks:  # init: each marked atom is a FG
        if atom2fg[atom] is None:
            fgs.append({atom})
            atom2fg[atom] = len(fgs) - 1
    
    for bond in mol.GetBonds():  # merge FGs
        a1, a2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if mol.GetAtomWithIdx(a1).GetSymbol() == 'H' or mol.GetAtomWithIdx(a2).GetSymbol() == 'H':
            continue
            
        if a1 in marks and a2 in marks:  
            # a marked atom should only belong to a FG, if atoms are both marked, merge their FGs into a FG
            fgs[atom2fg[a1]].update(fgs[atom2fg[a2]])
            fgs[atom2fg[a2]] = set()
            for a in fgs[atom2fg[a1]]:
                atom2fg[a] = atom2fg[a1]
    fgs = [fg for fg in fgs if len(fg) > 0]
    return fgs


def get_fg_set(
    mol, 
    verbose: bool = False, 
    custom_fgs: Dict[str, str] = dict(),
    **kwargs
):
    """
    Identify FGs and convert to SMILES
    Args:
        mol:
    Returns: a set of FG's SMILES
    """
    def vprint(*args, **kwargs):
        if verbose:
            print(*args, **kwargs)

    # find rings
    rings = find_rings(mol, **kwargs)
    vprint(f'Found Rings: {rings}')

    # find functional groups
    fgs = find_fgs(mol, custom_fgs)
    # merge fg with rings
    for ring in rings:
        for i in range(len(fgs)):
            if len(fgs[i].intersection(ring)) > 0:
                ring.update(fgs[i])
                fgs[i] = set()
    fgs = [fg for fg in fgs if len(fg) > 0]
    fg_atoms = []
    for fg in fgs:
        fg_atoms += list(fg)
    vprint(f'Found fgs: {fgs}')

    # get carbons
    carbons = []
    for idx in range(mol.GetNumAtoms()):
        if mol.GetAtomWithIdx(idx).GetSymbol() == 'H':
            continue
        if all([idx not in fg for fg in fgs + rings]):
            carbons.append(set([idx]))
    vprint(f'Found carbons: {carbons}')

    # functional groups neighbors, plus hydrogens
    all_fgs = fgs + rings + carbons
    for fg in all_fgs:
        hs = set()
        for at in fg:
            atom = mol.GetAtomWithIdx(at)
            hs.update(nei.GetIdx() for nei in atom.GetNeighbors() if nei.GetSymbol() == 'H')
        fg.update(hs)
    vprint(f"All functional groups: {all_fgs}")
    
    atom2fg = {}
    for i, fg in enumerate(all_fgs):
        for atom in fg:
            atom2fg[atom] = i
    fg_neis = [[] for _ in range(len(all_fgs))]
    for bond in mol.GetBonds():
        a1, a2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        fg1, fg2 = atom2fg[a1], atom2fg[a2]
        if fg1 != fg2:
            fg_neis[fg1].append(fg2)
            fg_neis[fg2].append(fg1)
    
    frags = {}
    for i in range(mol.GetNumAtoms()):
        if mol.GetAtomWithIdx(i).GetSymbol() != 'H':
            # collect its neighbor fgs
            fg_idx = atom2fg[i]
            fg_indices = [fg_idx] + fg_neis[fg_idx]
            fg = set()
            for idx in fg_indices:
                fg.update(all_fgs[idx])
            # cap with carbon
            caps = set()
            for at in fg:
                atom = mol.GetAtomWithIdx(at)
                if at in fg_atoms:
                    caps.update(nei.GetIdx() for nei in atom.GetNeighbors())
            fg.update(caps)
            
            frags[i] = fg
            for at in fg:
                for nei in mol.GetAtomWithIdx(at).GetNeighbors():
                    if nei.GetIdx() == 'H':
                        frags[nei.GetIdx()] = fg

    return frags


def cap_open_valences(fragment):
    obConversion = openbabel.OBConversion()
    obConversion.SetInAndOutFormats("mol", "mol")

    # Convert RDKit molecule to OpenBabel molecule
    mol_block = Chem.MolToMolBlock(fragment)
    ob_mol = openbabel.OBMol()
    obConversion.ReadString(ob_mol, mol_block)
    
    # Add hydrogens to OpenBabel molecule
    ob_mol.AddHydrogens()
    
    # Convert OpenBabel molecule back to RDKit molecule
    capped_mol_block = obConversion.WriteString(ob_mol)
    capped_mol = Chem.MolFromMolBlock(capped_mol_block, removeHs=False) #sanitize=False)
    
    return capped_mol


def select_fragment(mol, atomIndices):
    # de-sanitize
    dsmol = Chem.MolFromMolBlock(Chem.MolToMolBlock(mol), sanitize=False, removeHs=False)
    atomsToRemove = set(range(dsmol.GetNumAtoms())) - set(atomIndices)
    emol = Chem.EditableMol(dsmol)
    for atom in sorted(atomsToRemove, reverse=True):
        emol.RemoveAtom(atom)
    frag = emol.GetMol()
    Chem.SanitizeMol(
        frag,
        Chem.SANITIZE_ALL ^ Chem.SANITIZE_SETAROMATICITY ^ Chem.SANITIZE_KEKULIZE
    )
    Chem.SetAromaticity(frag, Chem.AromaticityModel.AROMATICITY_MDL) 
    frag = cap_open_valences(frag)
    return frag


def write_sdf(mol, sdf):
    writer = Chem.SDWriter(sdf)
    writer.write(mol)
    writer.close()


def label_mol(mol):
    for atom in mol.GetAtoms():
        atom.SetProp('atomLabel', f'{atom.GetSymbol()}:{atom.GetIdx()}')


def get_terminal_oxygens(atom):
    ter_os = []
    for nei in atom.GetNeighbors():
        if nei.GetSymbol() == 'O' and len(nei.GetNeighbors()) == 1:
            ter_os.append(nei)
    return ter_os


def fix_carboxylate(mol):
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == "C" and len(atom.GetNeighbors()) == 3:
            ter_os = get_terminal_oxygens(atom)
            if len(ter_os) == 2:
                atom.SetIsAromatic(False)
                ter_os[0].SetIsAromatic(False)
                ter_os[0].SetFormalCharge(0)
                ter_os[1].SetIsAromatic(False)
                bo1 = mol.GetBondBetweenAtoms(ter_os[0].GetIdx(), atom.GetIdx())
                bo1.SetIsAromatic(False)
                bo1.SetBondType(Chem.BondType.DOUBLE)
                
                ter_os[1].SetFormalCharge(-1)
                bo2 = mol.GetBondBetweenAtoms(ter_os[1].GetIdx(), atom.GetIdx())
                bo2.SetIsAromatic(False)
                bo2.SetBondType(Chem.BondType.SINGLE)
#         atom.UpdatePropertyCache()
    return mol


def fix_guanidine(mol):
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == 'C' and len(atom.GetNeighbors()) == 3:
            ns = [nei for nei in atom.GetNeighbors() if nei.GetSymbol() == 'N']
            
            num_ar_bonds = 0
            num_non_ring_ar_bonds = 0
            for i, n in enumerate(ns):
                bo = mol.GetBondBetweenAtoms(n.GetIdx(), atom.GetIdx())
                if bo.GetIsAromatic():
                    num_ar_bonds += 1
                    if not bo.IsInRing():
                        num_non_ring_ar_bonds += 1
            
            if num_ar_bonds >= 2 and num_non_ring_ar_bonds >= 1:
                atom.SetIsAromatic(False)
                flag = True
                for n in ns:
                    n.SetIsAromatic(False)
                    bo = mol.GetBondBetweenAtoms(n.GetIdx(), atom.GetIdx())
                    bo.SetIsAromatic(False)
                    if sum([a.GetSymbol() == 'H' for a in n.GetNeighbors()]) == 2 and flag:
                        n.SetFormalCharge(1)
                        bo.SetBondType(Chem.BondType.DOUBLE)
                        flag = False
                    else:
                        bo.SetBondType(Chem.BondType.SINGLE)
                        n.SetFormalCharge(0)
                assert (not flag), "Guandine is not fixed"
#         atom.UpdatePropertyCache()
    return mol


def read_pdb(pdb_path):
    mol = Chem.MolFromPDBFile(pdb_path, removeHs=False, sanitize=False)
    
#     obConversion = openbabel.OBConversion()
#     obConversion.SetInAndOutFormats("pdb", "mol")
    
#     ob_mol = openbabel.OBMol()
#     obConversion.ReadFile(ob_mol, pdb_path)
#     mol = Chem.MolFromMolBlock(obConversion.WriteString(ob_mol), removeHs=False, sanitize=False)
    
    fix_carboxylate(mol)
    fix_guanidine(mol)
    Chem.SanitizeMol(mol)
    return mol


def extract_info(mol):
    return {
        'natoms': int(mol.GetNumAtoms()),
        'charge': int(sum(at.GetFormalCharge() for at in mol.GetAtoms())),
        'elements': [at.GetSymbol() for at in mol.GetAtoms()],
        'xyz': mol.GetConformer().GetPositions(),
        'smiles': Chem.MolToSmiles(Chem.RemoveHs(mol), kekuleSmiles=True)
    }


def df2xyz(df, xyz):
    """
    Convert Dataframe to XYZ file, for visualization only!
    
    Parameters
    ----------
    df: pd.DataFrame object
        Must have a column called 'xyz' containing 3D coordinates
        and a column 'elements' containing atom symbols
    xyz: os.PathLike
        The path to the written XYZ file
    """
    fp = open(xyz, 'w')
    for xyz, elements, index in zip(df['xyz'], df['elements'], df.index):
        elements = elements.split()
        xyz = xyz.split()
        xyz = np.array([float(x) for x in xyz]).reshape(-1, 3)
        xyz -= np.mean(xyz, axis=0)
        fp.write(f'{len(elements)}\n{index}\n')
        for i in range(len(elements)):
            fp.write(f"{elements[i]} {xyz[i, 0]:.4f} {xyz[i, 1]:.4f} {xyz[i, 2]:.4f}\n")
    fp.close()