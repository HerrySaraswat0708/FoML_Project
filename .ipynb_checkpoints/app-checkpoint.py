import streamlit as st
import pandas as pd
import numpy as np
import joblib

from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import rdFingerprintGenerator
from rdkit.DataStructs import ConvertToNumpyArray

# -------------------------------
# Load Data + Model
# -------------------------------
df = pd.read_csv("curated-solubility-dataset.csv")
model = joblib.load("best_solubility_model.pkl")
pipeline = joblib.load("final_pipeline.pkl")
features = pipeline['features']


# -------------------------------
# Fingerprint Generator
# -------------------------------
fp_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)

def smiles_to_vector(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    fp = fp_gen.GetFingerprint(mol)
    arr = np.zeros((1024,), dtype=int)
    ConvertToNumpyArray(fp, arr)
    
    return arr

# -------------------------------
# Interpretation Function
# -------------------------------
def interpret_solubility(logS):
    if logS > -1:
        return "Highly Soluble 🟢"
    elif logS > -3:
        return "Soluble 🟡"
    elif logS > -5:
        return "Moderately Soluble 🟠"
    else:
        return "Poorly Soluble 🔴"

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("🔬 Molecular Solubility Predictor")

st.write("Select a molecule to predict its solubility.")

# Dropdown
selected_index = st.selectbox(
    "Choose Molecule (by index)",
    df.index
)

# Get SMILES
smiles = df.loc[selected_index, "SMILES"]

# -------------------------------
# Display Molecule
# -------------------------------
mol = Chem.MolFromSmiles(smiles)

st.subheader("🧪 Molecule Structure")
if mol:
    img = Draw.MolToImage(mol)
    st.image(img, caption=smiles)
else:
    st.error("Invalid SMILES")

# -------------------------------
# Prediction
# -------------------------------
if st.button("Predict Solubility"):

    # Fingerprint
    fp = smiles_to_vector(smiles)
    
    if fp is None:
        st.error("Cannot process molecule")
    else:
        # Descriptor features
        desc = df.loc[selected_index, feature_names].values
        
        # Combine
        X_input = np.concatenate([fp, desc]).reshape(1, -1)
        
        # Predict
        logS = model.predict(X_input)[0]
        sol = 10 ** logS
        category = interpret_solubility(logS)
        
        # Output
        st.subheader("📊 Prediction Results")
        st.write(f"**LogS:** {logS:.2f}")
        st.write(f"**Solubility:** {sol:.6f} mol/L")
        st.write(f"**Category:** {category}")