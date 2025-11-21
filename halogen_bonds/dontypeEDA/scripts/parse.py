from typing import List, Dict, Any, Optional
import os
import numpy as np


def opt_geoemtry(fname: os.PathLike):
    atoms = []
    coords = []
    read = False
    with open(fname) as f:
        for line in f:
            line = line.strip()
            if line.startswith('** TRANSITION STATE CONVERGED  **'):
                read = True
            if line.startswith('**  OPTIMIZATION CONVERGED  **'):
                read = True
            if line.startswith('Molecular Point Group'):
                read = False
            if line.startswith('Z-matrix Print:'):
                read = False

            if read:
                content = line.split()
                if len(content) == 0:
                    continue
                if content[0].isdigit():
                    atom = str(content[1])
                    x = float(content[2])
                    y = float(content[3])
                    z = float(content[4])
                    atoms.append(atom)
                    coords.append([x, y, z])
                    # coords['atom'].append(str(content[1]))
                    # coords['X'].append(float(content[2])) 
                    # coords['Y'].append(float(content[3])) 
                    # coords['Z'].append(float(content[4]))
    return atoms, coords


def pesscan_geometry(fname: os.PathLike):
    coords_list = []
    coords = {'atom': [],
             'X': [],
             'Y': [], 
             'Z': []}
    read = False
    with open(fname) as f:
        
        for line in f:
            line = line.strip()
            if line.startswith('**  OPTIMIZATION CONVERGED  **'):
                read = True
            if line.startswith('Z-matrix Print:'):
                read = False
                coords_list.append(coords)
                coords = {key: [] for key in coords}

            if read:
                content = line.split()
                if len(content) == 0:
                    continue
                if content[0].isdigit():
                    coords['atom'].append(str(content[1]))
                    coords['X'].append(float(content[2])) 
                    coords['Y'].append(float(content[3])) 
                    coords['Z'].append(float(content[4]))
    return coords_list


def find_mp2eda_error(fname: os.PathLike):
    read = False
    error = False
    with open(fname) as f:
        for line in f:
            line = line.strip()
            if line.startswith("Q-Chem fatal error"):
                error = True

    return error


def sp_output(fname: os.PathLike):
    read = False
    with open(fname) as f:
        for line in f:
            line = line.strip()
            if line.startswith("Components of the RIMP2 correlation energy:"):
                read = True
                continue 

            if read:
                content = line.split()
                if len(content) == 0:
                    continue
                if content[0] == 'RIMP2' and content[1] == 'total':
                    energy = float(content[4])
                    read = False
                    break
    return energy


def opt_energy_output_2(fname: os.PathLike):
    read = False
    point = []
    with open(fname) as f:
        for line in f:
            line = line.strip()

            if line.startswith('$opt'):
                read = True
                continue

            if line.startswith('$end'):
                read = False
                continue

            if line.startswith("Minimum search - taking simple RFO step"):
                read = True
                continue
        
            if line.startswith("Minimum search - taking P-RFO step"):
                read = True
                continue
                
            if line.startswith("Z-matrix Print:"):
                read = True
                continue

            if line.startswith("**  OPTIMIZATION CONVERGED  **"):
                read = False
                
            if read:
                content = line.split()
                if len(content) == 0:
                    continue
                if content[0] == 'Final' and content[1] == 'energy':
                    point.append(float(content[3]))
                if content[0] == 'stre':
                    point.append(float(content[3]))
    return [point]


def energy_output(fname: os.PathLike):
    read = False
    with open(fname) as f:
        for line in f:
            line = line.strip()
            if line.startswith("User input:"):
                read = True
                continue

            if read:
                content = line.split()
                if len(content) == 0:
                    continue
                if content[0] == 'JOBTYPE':
                    if content[1] == 'pes_scan':
                        PesScan = True
                        break
                    if content[1] == 'opt':
                        PesScan = False
                        break
    if PesScan:
        return pesscan_output(fname)
    else:
        return [opt_energy_output(fname)]
           

def pesscan_output(fname: os.PathLike):
    read = False
    Pes = []
    with open(fname) as f:
        for line in f:
            line = line.strip()

            if line.startswith("------- Summary of potential scan: ------"):
                read = True
                continue

            if line.startswith("-----------------------------------------"):
                read = False
                continue

            if read:
                content = line.split()
                Pes.append([float(content[0]), float(content[1])])
    return Pes

def opt_energy_output(fname: os.PathLike):
    read = False
    point = []
    with open(fname) as f:
        for line in f:
            line = line.strip()

            if line.startswith("Distance Matrix (Angstroms)"):
                read = True
                continue

            if line.startswith("**  OPTIMIZATION CONVERGED  **"):
                read = False
                
            if read:
                content = line.split()
                if len(content) == 0:
                    continue
                if content[0] == 'Final' and content[1] == 'energy':
                    point.append(float(content[3]))
                if content[0] == "Cl(" and len(content) == 4:
                    point.append(float(content[3]))
    return point


def eda_output(fname: os.PathLike, unit: str = 'kJ') -> Dict[str, float]:
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
        "PREP": None,
        "FRZ": None,
        "POL": None,
        "CT": None,
        "INTERACTION": None,
        "ELEC": None,
        "MOD_PAULI": None,
        "DISP": None,
        "TOTAL": None
    }
    _read = False
    with open(fname, 'r') as f:
        for line in f:
            line = line.strip()
            
            if line.startswith("Results of EDA2"):
                _read = True
                continue
            if line.startswith("Decomposition of frozen interaction energy"):
                _read = True
                continue

            if _read:
                content = line.split()
                if len(content) == 0:
                    continue
                if content[0] == "E_prp":
                    res['PREP'] = float(content[-1])
                elif content[0] == "E_frz":
                    res['FRZ'] = float(content[-1])
                elif content[0] == "E_pol":
                    res["POL"] = float(content[-1])
                elif content[0] == "E_vct":
                    res['CT'] = float(content[-1])
                elif content[0] == "E_int":
                    res['INTERACTION'] = float(content[-1])
                elif content[0] == "E_cls_elec":
                    res['ELEC'] = float(content[-1])
                elif content[0] == "E_mod_pauli" or content[0] == '[E_mod_pauli':
                    res['MOD_PAULI'] = float(content[5])
                elif content[0] == "E_cls_disp" or content[0] == "E_disp":
                    res['DISP'] = float(content[-1])
                elif content[0] == "TOTAL":
                    res["TOTAL"] = float(content[1])
                    _read = False
                    break

    
    if unit == "kcal":
        for key in res.keys():
            res[key] /= 4.184
    return res


def mp2_output(fname: os.PathLike, unit: str = 'kJ') -> Dict[str, float]:
    """
    Parse the QChem MP2 EDA job output
    
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
        "Dist": None,
        "FRZ": None,
        "POL": None,
        "DISP": None,
        "DISP_FULL": None,
        "CT": None,
        "INTERACTION": None,
        "DISP_MP2": None,
        "DISP_MP2_FULL": None,
        "CT_MP2_REL_DISP": None,
        "CT_MP2_REL_DISP_FULL": None
    }
    _read_EDA = False
    _read_MP2 = False
    _read_dist = False
    with open(fname, 'r') as f:
        for line in f:
            line = line.strip()

            if line.startswith("Distance Matrix (Angstroms)"):
                _read_dist = True
                continue
            
            if line.startswith("RELAVENT ENERGIES (kJ/mol)"):
                _read_EDA = True
                continue

            if line.startswith("SOME ENERGIES (kJ/mol)"):
                _read_MP2 = True
                continue

            if _read_dist:
                content = line.split()
                if len(content) == 0:
                    continue
                if content[0] == "Cl(" and len(content) == 4:
                    res['Dist'] = float(content[3])
                    _read_dist = False
                    

            if _read_EDA:
                content = line.split()
                if len(content) == 0:
                    continue
                if content[0] == "FRZ":
                    res['FRZ'] = float(content[-1])
                elif content[0] == "POL":
                    res['POL'] = float(content[-1])
                elif content[0] == "DISP":
                    res["DISP"] = float(content[-1])
                elif content[0] == "DISP-FULL":
                    res['DISP_FULL'] = float(content[-1])
                elif content[0] == "CT":
                    res['CT'] = float(content[-1])
                elif content[0] == "INTERACTION":
                    res["INTERACTION"] = float(content[-1])
                    _read_EDA = False

            if _read_MP2:
                content = line.split()
                if len(content) == 0:
                    continue
                if content[0] == "DISP":
                    res['DISP_MP2'] = float(content[-1])
                elif content[0] == "DISP-FULL":
                    res['DISP_MP2_FULL'] = float(content[-1])
                elif content[0] == "DISP-FULL":
                    res['DISP_FULL'] = float(content[-1])
                elif content[0] == "CT" and content[4] == "disp)":
                    res['CT_MP2_REL_DISP'] = float(content[-1])
                elif content[0] == "CT" and content[4] == "disp-full)":
                    res["CT_MP2_REL_DISP_FULL"] = float(content[-1])
                    _read_MP2 = False
                    break
    
    if unit == "kcal":
        for key in res.keys():
            res[key] /= 4.184
    return res
