import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.DataStructs import ConvertToNumpyArray
from rdkit import RDLogger
from rdkit.Chem import Draw

RDLogger.DisableLog('rdApp.*')


df = pd.read_csv('curated-solubility-dataset.csv')
smiles_list = df["SMILES"] 

fp_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=1024)
def smiles_to_vector(smiles):
    mol = Chem.MolFromSmiles(smiles)
    
    if mol is None:
        return None  
    
    fp = fp_gen.GetFingerprint(mol)
    
    arr = np.zeros((1024,), dtype=int)
    ConvertToNumpyArray(fp, arr)
    
    return arr


vectors = []
valid_indices = []

for i, smi in enumerate(smiles_list):
    vec = smiles_to_vector(smi)
    
    if vec is not None:
        vectors.append(vec)
        valid_indices.append(i)

X_fp = np.array(vectors)



df_clean = df.iloc[valid_indices].reset_index(drop=True)

numeric_df = df_clean.select_dtypes(include=['float64', 'int64'])

corr_with_target = numeric_df.corr()["Solubility"].sort_values(ascending=False)

print(corr_with_target)

important_features = corr_with_target[abs(corr_with_target) > 0.1]
print(important_features)
important_features = important_features.drop("Solubility")

X_feat = df_clean[important_features.index]
X = np.concatenate([X_fp, X_feat], axis=1)
y = np.array(df_clean['Solubility'])

np.save('X.npy',X)
np.save('y.npy',y)
