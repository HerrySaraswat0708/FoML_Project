import networkx as nx
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger
import torch
from torch_geometric.data import Data
# import matplotlib.pyplot as plt
RDLogger.DisableLog('rdApp.*')


# def clean_filename(name):
#     name = re.sub(r'[^\w\-_. ]', '', name)   
#     return name[:50]  

df = pd.read_csv('data\\curated-solubility-dataset.csv')

# print(df)
# def get_bond_weight(bond):
#     btype = bond.GetBondType()
    
#     if btype.name == "SINGLE":
#         return 1
#     elif btype.name == "DOUBLE":
#         return 2
#     elif btype.name == "TRIPLE":
#         return 3
#     else:
#         return 1.5
    
# graphs = []

# for _, row in df.iterrows():
#     mol = Chem.MolFromSmiles(row['SMILES'])
#     if mol:
#         G = nx.Graph()

#         for atom in mol.GetAtoms():
#             G.add_node(atom.GetIdx(), symbol=atom.GetSymbol())
#             # atomic_num=atom.GetAtomicNum()

#         for bond in mol.GetBonds():
#             G.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), weight=get_bond_weight(bond))

#         G.graph['label'] = row['Solubility']
#         safe_name = clean_filename(row['Name'])
#         nx.write_graphml(G, f"graph-data\{safe_name}.graphml")
#         graphs.append(G)



def atom_to_feature(atom):
    return [
        atom.GetAtomicNum(),        
        atom.GetDegree(),          
        int(atom.GetIsAromatic()),  
    ]



def mol_to_graph(smiles, label):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    # Nodes
    x = []
    for atom in mol.GetAtoms():
        x.append(atom_to_feature(atom))
    x = torch.tensor(x, dtype=torch.float)

    # Edges
    edge_index = []
    edge_attr = []

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        # undirected graph → add both directions
        edge_index.append([i, j])
        edge_index.append([j, i])

        # bond weight
        btype = bond.GetBondType().name
        if btype == "SINGLE":
            w = 1
        elif btype == "DOUBLE":
            w = 2
        elif btype == "TRIPLE":
            w = 3
        else:
            w = 1.5

        edge_attr.append([w])
        edge_attr.append([w])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    y = torch.tensor([label], dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

dataset = []

for _, row in df.iterrows():
    print(row)
    data = mol_to_graph(row['SMILES'], row['Solubility'])
    if data is not None:
        dataset.append(data)

torch.save(dataset, "data\\aqsoldb_graph_dataset.pt")





# Draw.MolToImage(Chem.MolFromSmiles(smiles))









# idx = 10
# G = graphs[idx]
# molecule_name = df.loc[idx]['Name']
# print(molecule_name)
# pos = nx.spring_layout(G)


# node_labels = nx.get_node_attributes(G, 'symbol')
# edge_labels = nx.get_edge_attributes(G, 'weight')

# nx.draw(G, pos, with_labels=True, labels=node_labels, node_color='lightblue')
# nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
# plt.title(f"Molecule: {molecule_name}")
# plt.subplots_adjust(top=0.85)
# plt.tight_layout()
# plt.show()