import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))
from edautils.qchem import parse_qchem_eda_output
import pandas as pd
from tqdm import tqdm


directories = ['hbonds', 'halogen_bonds', 'pi_stack', 'pi_cation', 'salt_bridge', 'hydrophobic']
csvs = ['hbond_EDA.csv', 'halogen_EDA.csv', 'pi_stack_EDA.csv', 'pi_cation_EDA.csv', 'salt_bridge_EDA.csv', 'hydrophobic_EDA.csv']

for fcsv, dirname in zip(csvs, directories):
    df = pd.read_csv(os.path.join(dirname, fcsv), index_col=0)
    if dirname == 'pi_stack':
        eda_dir = '/pscratch/sd/e/eric6/BioEdaDatabase-joey/pi_stack/pi_stack_EDA_wb97mv/pi_stack_EDA'
    else:
        eda_dir = os.path.join(dirname, [x for x in os.listdir(dirname) if x.endswith('EDA')][0])
    for index in tqdm(df.index, desc=dirname):
        res = parse_qchem_eda_output(os.path.join(eda_dir, f'{index}/eda.out'))
        if index.startswith("4z1f_108_PHE"):
            print(res)
        for key in res:
            df.loc[index, key] = res[key]
    df.to_csv(os.path.join(dirname, fcsv))
        
