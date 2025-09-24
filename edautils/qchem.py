from typing import List, Dict, Any, Optional
import os
import numpy as np
import parmed


DEFAULT_QCHEM_SETTING = {
    "JOBTYPE": "eda",
    "EDA2": 1,
    "METHOD": "wB97M-V",
    "BASIS": "def2-QZVPPD",
    "SCF_CONVERGENCE": 8,
    "THRESH": 14,
    "XC_GRID": "000099000590",
    "NL_GRID": 1,
    "SYMMETRY": False,
    "EDA_BSSE": False,
    "FD_MAT_VEC_PROD": False,
    "MEM_TOTAL": 8000,
    "MEM_STATIC": 1000,
    "SCF_PRINT_FRGM": True
}


def mk_qchem_eda_input(
    atoms: List[List[str]], 
    coords: List[np.ndarray],
    charges: List[int],
    names: List[str],
    fname: Optional[os.PathLike] = None,
    setting: Dict[str, Any] = {}
):
    """
    Make a QChem EDA job input file

    Parameters
    ----------
    atoms: List[List[str]]
        Atoms of each fragment
    coords: list of np.ndarray
        Coordinates of each fragment
    charges: list of int
        Net charges of the fragments
    names: list of str
        Names of the fragments
    fname: os.PathLike, optional
        Path to the output qchem job input file
    setting: dict
        Additonal settings that overwrites the `DEFAULT_QCHEM_SETTING`

    Returns
    -------
    inpstr: str
        The content of the input file
    """
    inpstr = [f"$molecule\n{sum(charges)} 1\n"]
    for i in range(len(atoms)):
        inpstr.append(f"-- {names[i]}\n{charges[i]} 1\n")
        for at, crd in zip(atoms[i], coords[i]):
            inpstr.append(f"{at:>3} {crd[0]:>13.7f} {crd[1]:>13.7f} {crd[2]:>13.7f}\n")
    inpstr.append("$end\n\n")
    inpstr.append("$rem\n")

    tmpsetting = DEFAULT_QCHEM_SETTING.copy()
    tmpsetting.update(setting)
    maxlen = max([len(key) for key in tmpsetting.keys()])
    tmpl = "   {:<" + str(maxlen) + "}  =  {}\n"
    for key, value in tmpsetting.items():
        if isinstance(value, bool): value = str(value).lower()
        inpstr.append(tmpl.format(key.upper(), value))
    inpstr.append("$end\n\n")
    inpstr = "".join(inpstr)
    if fname:
        with open(fname, 'w') as f:
            f.write(inpstr)
    return inpstr


def parse_qchem_eda_output(fname: os.PathLike, unit: str = 'kJ') -> Dict[str, float]:
    """
    Parse the QChem EDA job output
    
    Parameters
    ----------
    fname: os.PathLike
        Path to the output file
    unit: str
        Energy unit, default is 'kJ` (kJ/mol). Users can switch it to 'kcal'
    
    Returns
    -------
    res: dict
        A python dictionary in which keys are enery terms and values are corresponding energy values
    """
    res = {
        "ELEC": None,
        "PAULI": None,
        "DISP": None,
        "CLS_ELEC": None,
        "MOD_PAULI": None,
        "CLS_DISP": None,
        "PREPARATION": None,
        "FROZEN": None,
        "POLARIZATION": None,
        "CHARGE_TRANSFER": None,
        "TOTAL": None
    }
    _read = False
    old_version = None
    with open(fname, 'r') as f:
        for line in f:
            line = line.strip()

            if (old_version is None) and line.startswith("Q-Chem"):
                if int(line.split()[1].split('.')[0]) < 6:
                    old_version = True
                else:
                    old_version = False
            
            if (not old_version) and line.startswith("Decomposition of frozen interaction energy"):
                _read = True
                continue
            if (old_version) and line.startswith("Initial Wavefunction Decomposition"):
                _read = True
                continue

            if _read:
                content = line.split()
                if len(content) == 0:
                    continue
                if content[0] == "E_elec":
                    res['ELEC'] = float(content[-1])
                elif content[0] == "E_pauli":
                    res['PAULI'] = float(content[-1])
                elif content[0] == "E_disp":
                    res["DISP"] = float(content[-1])
                elif content[0] == "E_cls_elec":
                    res['CLS_ELEC'] = float(content[-1])
                elif content[0] == "E_mod_pauli" or content[0] == '[E_mod_pauli':
                    res['MOD_PAULI'] = float(content[5])
                elif content[0] == "E_cls_disp":
                    res['CLS_DISP'] == float(content[-1])
                elif content[0] == "PREPARATION":
                    res["PREPARATION"] = float(content[-1])
                elif content[0] == "FROZEN":
                    res["FROZEN"] = float(content[1])
                    if old_version:
                        res['FROZEN'] += res['DISP']
                elif content[0] == "POLARIZATION":
                    res['POLARIZATION'] = float(content[-1])
                elif content[0] == "CHARGE":
                    res['CHARGE_TRANSFER'] = float(content[-1])
                elif content[0] == "TOTAL":
                    res["TOTAL"] = float(content[1])
                    _read = False
                    break
    
    # assert abs(res["ELEC"] + res["PAULI"] - res["CLS_ELEC"] - res["MOD_PAULI"]) < 2e-4
    # frozen_error = abs(res["FROZEN"] - res["CLS_ELEC"] - res['MOD_PAULI'] - res['DISP'])
    # assert frozen_error < 2e-4, f"{frozen_error} too large"
    
    # total_error = abs(res['TOTAL'] - res['PREPARATION'] - res['FROZEN'] - res['CHARGE_TRANSFER'] - res['POLARIZATION'])
    # assert total_error < 5e-4, f"{total_error} too large"
    
    if unit == "kcal":
        for key in res.keys():
            res[key] /= 4.184
    return res


def parse_qchem_opt_output(fname: os.PathLike):
    read = 0
    pos = []
    symbols = []
    with open(fname) as f:
        for line in f:
            if read == 0:
                read = line.strip().startswith("**  OPTIMIZATION CONVERGED  **")
            elif read >=1 and read < 6:
                read += 1
            elif read >= 6:
                content = line.strip().split()
                if content[0].startswith("-"):
                    read = 0
                    break
                else:
                    pos.append([float(content[-3]), float(content[-2]), float(content[-1])])
                    symbols.append(content[1])
    pos = np.array(pos)
    return pos, symbols


def to_pdb(coords, symbols, fname, resname="UNK"):
    struct = parmed.Structure()
    res = parmed.Residue(name=resname)
    struct.residues.append(res)
    count = {}
    for coord, symb in zip(coords, symbols):
        cnt = count.get(symb, 0) + 1
        count[symb] = cnt
        atname = f"{symb}{cnt}"
        atom = parmed.Atom(
            atomic_number=parmed.periodic_table.AtomicNum[symb],
            name=atname
        )
        atom.xx, atom.xy, atom.xz = tuple(map(float, coord))
        struct.add_atom(atom, resname, -1)
    struct.save(fname, overwrite=True, use_hetatoms=False)


def qchem_opt_output_to_pdb(path_qchem_output, path_pdb, resname="UNK"):
    coords, symbols = parse_qchem_opt_output(path_qchem_output)
    to_pdb(coords, symbols, path_pdb, resname)


def parse_qchem_sp_output(fname: os.PathLike):
    read = 0
    pos = []
    symbols = []
    with open(fname) as f:
        for line in f:
            if read == 0:
                read = line.strip().startswith("Standard Nuclear Orientation")
            elif read >=1 and read < 3:
                read += 1
            elif read >= 3:
                content = line.strip().split()
                if content[0].startswith("-"):
                    read = 0
                    break
                else:
                    pos.append([float(content[-3]), float(content[-2]), float(content[-1])])
                    symbols.append(content[1])
    pos = np.array(pos)
    return pos, symbols