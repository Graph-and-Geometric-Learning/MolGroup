import os
import random
import numpy as np
from tqdm import tqdm
from time import time

import torch
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem


def smile2fp(smile):
    mol = AllChem.MolFromSmiles(smile)
    try:
        # return AllChem.GetMACCSKeysFingerprint(mol)
        return AllChem.GetMorganFingerprintAsBitVect(mol, 2)
    except:
        return None


def getmorganfingerprint(mol):
    # return list(AllChem.GetMorganFingerprintAsBitVect(mol, 2))
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2)


def getmaccsfingerprint(mol):
    fp = AllChem.GetMACCSKeysFingerprint(mol)
    # return [int(b) for b in fp.ToBitString()]
    return fp


def extract_fp(smiles):
    mgf_feats = []
    maccs_feats = []
    fps = []
    for ii in range(len(smiles)):
        if isinstance(smiles, list):
            rdkit_mol = AllChem.MolFromSmiles(smiles[ii])
        else:
            rdkit_mol = AllChem.MolFromSmiles(smiles.iloc[ii])

        mgf = getmorganfingerprint(rdkit_mol)
        mgf_feats.append(mgf)

        maccs = getmaccsfingerprint(rdkit_mol)
        maccs_feats.append(maccs)
        
        fp = Chem.RDKFingerprint(rdkit_mol)
        fps.append(fp)

    # mgf_feats = np.array(mgf_feats, dtype="int64")
    # maccs_feats = np.array(maccs_feats, dtype="int64")
    # fps = np.array(fps, dtype="int64")

    return mgf_feats, maccs_feats, fps


def get_fingerprint(dataset_name, data_path, fp_type="mgf"):
    if dataset_name != "qm8" and dataset_name != "qm9":
        save_path = os.path.join(data_path, f"{dataset_name}/mapping/".replace("-", "_"))
    else:
        save_path = os.path.join(data_path, f"{dataset_name}/".replace("-", "_"))
    
    fp_save_path = os.path.join(save_path, f"fp_feats_{fp_type}_tensor.pt".replace("-", "_"))

    if not os.path.exists(fp_save_path):
        if dataset_name != "qm8" and dataset_name != "qm9":
            smile_path = os.path.join(data_path, f"{dataset_name}/mapping/mol.csv.gz".replace("-", "_"))
            df_smi = pd.read_csv(smile_path)
            smiles = df_smi["smiles"]
        else:
            raw_dataset = pd.read_csv(os.path.join(data_path, dataset_name, dataset_name + ".csv"))
            smiles = [smile for smile in raw_dataset['smiles'].tolist()]

        mgf_feat, maccs_feat, fps = extract_fp(smiles)
        mgf_feat = torch.tensor(mgf_feat).float()
        maccs_feat = torch.tensor(maccs_feat).float()
        fps = torch.tensor(fps).float()

        torch.save(mgf_feat, os.path.join(save_path, f"fp_feats_mgf_tensor.pt".replace("-", "_")))
        torch.save(maccs_feat, os.path.join(save_path, f"fp_feats_macc_tensor.pt".replace("-", "_")))
        torch.save(fps, os.path.join(save_path, f"fp_feats_fp_tensor.pt".replace("-", "_")))

    feat = torch.load(fp_save_path)
    return feat
