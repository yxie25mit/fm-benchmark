import os
import csv
import math
import time
import random
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler

from torch_scatter import scatter
from torch_geometric.data import Data, Dataset, DataLoader

import rdkit
from rdkit import Chem
from rdkit.Chem.rdchem import HybridizationType
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
from rdkit import RDLogger                                                                                                                                                               
RDLogger.DisableLog('rdApp.*')  


ATOM_LIST = list(range(1,119))
CHIRALITY_LIST = [
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    Chem.rdchem.ChiralType.CHI_OTHER
]
BOND_LIST = [BT.SINGLE, BT.DOUBLE, BT.TRIPLE, BT.AROMATIC, BT.DATIVE]
BONDDIR_LIST = [
    Chem.rdchem.BondDir.NONE,
    Chem.rdchem.BondDir.ENDUPRIGHT,
    Chem.rdchem.BondDir.ENDDOWNRIGHT
]

# Dict lookups replace the list.index() linear scans in the per-atom/per-bond loops.
ATOM_INDEX = {value: i for i, value in enumerate(ATOM_LIST)}
CHIRALITY_INDEX = {value: i for i, value in enumerate(CHIRALITY_LIST)}
BOND_INDEX = {value: i for i, value in enumerate(BOND_LIST)}
BONDDIR_INDEX = {value: i for i, value in enumerate(BONDDIR_LIST)}


def _generate_scaffold(smiles, include_chirality=False):
    mol = Chem.MolFromSmiles(smiles)
    scaffold = MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)
    return scaffold


def generate_scaffolds(dataset, log_every_n=1000):
    scaffolds = {}
    data_len = len(dataset)
    print(data_len)

    print("About to generate scaffolds")
    for ind, smiles in enumerate(dataset.smiles_data):
        if ind % log_every_n == 0:
            print("Generating scaffold %d/%d" % (ind, data_len))
        scaffold = _generate_scaffold(smiles)
        if scaffold not in scaffolds:
            scaffolds[scaffold] = [ind]
        else:
            scaffolds[scaffold].append(ind)

    # Sort from largest to smallest scaffold sets
    scaffolds = {key: sorted(value) for key, value in scaffolds.items()}
    scaffold_sets = [
        scaffold_set for (scaffold, scaffold_set) in sorted(
            scaffolds.items(), key=lambda x: (len(x[1]), x[1][0]), reverse=True)
    ]
    return scaffold_sets


def scaffold_split(dataset, valid_size, test_size, seed=None, log_every_n=1000):
    train_size = 1.0 - valid_size - test_size
    scaffold_sets = generate_scaffolds(dataset)

    train_cutoff = train_size * len(dataset)
    valid_cutoff = (train_size + valid_size) * len(dataset)
    train_inds: List[int] = []
    valid_inds: List[int] = []
    test_inds: List[int] = []

    print("About to sort in scaffold sets")
    for scaffold_set in scaffold_sets:
        if len(train_inds) + len(scaffold_set) > train_cutoff:
            if len(train_inds) + len(valid_inds) + len(scaffold_set) > valid_cutoff:
                test_inds += scaffold_set
            else:
                valid_inds += scaffold_set
        else:
            train_inds += scaffold_set
    return train_inds, valid_inds, test_inds


def read_smiles(data_path, target, task):
    """
    MT patch: `target` may now be either a single column name (str, original behavior)
    or a list of column names. Returns labels as list[float] (single-target, original)
    or list[list[float]] (multi-target). Missing entries become float('nan') in MT mode
    so downstream NaN-masked loss/eval can skip them per-target.
    """
    target_list = [target] if isinstance(target, str) else list(target)
    multi = len(target_list) > 1
    smiles_data, labels = [], []
    with open(data_path) as csv_file:
        csv_reader = csv.DictReader(csv_file, delimiter=',')
        for i, row in enumerate(csv_reader):
            # MT patch: original `if i != 0` was a bug — DictReader already skipped the
            # header row, so this dropped the FIRST data row. Harmless in single-target
            # mode (internal scaffold split sees the same shortened list), but breaks
            # external split indices computed on the full CSV. Removed the gate.
            if True:
                smiles = row['smiles']
                mol = Chem.MolFromSmiles(smiles)
                if mol is None and not multi:
                    continue
                if multi:
                    # MT patch: do NOT skip rows with invalid SMILES here (cleaned input
                    # is pre-validated; skipping shifts indices out of alignment with our
                    # pre-computed split files).  Just record raw row + NaN-filled labels.
                    row_labels = []
                    for t in target_list:
                        v = row.get(t, '')
                        row_labels.append(float('nan') if v in ('', None) else float(v))
                    smiles_data.append(smiles)
                    labels.append(row_labels)
                else:
                    if mol is None:
                        continue   # original single-target behavior
                    label = row[target_list[0]]
                    if label == '':
                        continue
                    smiles_data.append(smiles)
                    if task == 'classification':
                        # MT patch: cleaned CSVs store labels as floats ("1.0"),
                        # so int(label) fails. Coerce via float first.
                        labels.append(int(float(label)))
                    elif task == 'regression':
                        labels.append(float(label))
                    else:
                        ValueError('task must be either regression or classification')
    print(len(smiles_data))
    return smiles_data, labels


class MolTestDataset(Dataset):
    def __init__(self, data_path, target, task):
        super(Dataset, self).__init__()
        self.smiles_data, self.labels = read_smiles(data_path, target, task)
        self.task = task

        self.conversion = 1
        if 'qm9' in data_path and target in ['homo', 'lumo', 'gap', 'zpve', 'u0']:
            self.conversion = 27.211386246
            print(target, 'Unit conversion needed!')

        # The graph built from a SMILES is deterministic, so build it once instead of
        # re-running RDKit on every access of every epoch. Safe to hand out the cached
        # object: Batch.from_data_list copies every tensor (`item + cum` / `torch.cat`),
        # so nothing downstream can write through to the cache.
        self.data_cache = {}

    def precompute(self, indices):
        """Build and cache the graphs the loaders will touch, before workers fork."""
        for index in indices:
            if index not in self.data_cache:
                self.data_cache[index] = self._build_data(index)

    def __getitem__(self, index):
        data = self.data_cache.get(index)
        if data is None:
            data = self._build_data(index)
            self.data_cache[index] = data
        return data

    def _build_data(self, index):
        mol = Chem.MolFromSmiles(self.smiles_data[index])
        mol = Chem.AddHs(mol)

        N = mol.GetNumAtoms()
        M = mol.GetNumBonds()

        type_idx = []
        chirality_idx = []
        atomic_number = []
        for atom in mol.GetAtoms():
            type_idx.append(ATOM_INDEX[atom.GetAtomicNum()])
            chirality_idx.append(CHIRALITY_INDEX[atom.GetChiralTag()])
            atomic_number.append(atom.GetAtomicNum())

        x1 = torch.tensor(type_idx, dtype=torch.long).view(-1,1)
        x2 = torch.tensor(chirality_idx, dtype=torch.long).view(-1,1)
        x = torch.cat([x1, x2], dim=-1)

        row, col, edge_feat = [], [], []
        for bond in mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            row += [start, end]
            col += [end, start]
            bond_feat = [BOND_INDEX[bond.GetBondType()], BONDDIR_INDEX[bond.GetBondDir()]]
            edge_feat.append(list(bond_feat))
            edge_feat.append(list(bond_feat))

        edge_index = torch.tensor([row, col], dtype=torch.long)
        edge_attr = torch.tensor(np.array(edge_feat), dtype=torch.long)
        # MT patch: always store y as float (NaN-mask requires float).
        # Single-target cls case: 0/1 floats work identically with BCE.
        # Multi-target case: self.labels[index] is a list[float] possibly containing NaN.
        if self.task == 'classification':
            y = torch.tensor(self.labels[index], dtype=torch.float).view(1, -1)
        elif self.task == 'regression':
            arr = np.asarray(self.labels[index], dtype=np.float32) * self.conversion
            y = torch.tensor(arr, dtype=torch.float).view(1, -1)
        data = Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_attr)
        return data

    def __len__(self):
        return len(self.smiles_data)


class MolTestDatasetWrapper(object):
    
    def __init__(self,
        batch_size, num_workers, valid_size, test_size,
        data_path, target, task, splitting,
        external_split=None,  # MT patch: optional (train_idx, val_idx, test_idx) tuple
    ):
        super(object, self).__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.valid_size = valid_size
        self.test_size = test_size
        self.target = target
        self.task = task
        self.splitting = splitting
        self.external_split = external_split
        assert splitting in ['random', 'scaffold', 'external']

    def get_data_loaders(self):
        train_dataset = MolTestDataset(data_path=self.data_path, target=self.target, task=self.task)
        train_loader, valid_loader, test_loader = self.get_train_validation_data_loaders(train_dataset)
        return train_loader, valid_loader, test_loader

    def get_train_validation_data_loaders(self, train_dataset):
        if self.splitting == 'random':
            num_train = len(train_dataset)
            indices = list(range(num_train))
            np.random.shuffle(indices)
            split = int(np.floor(self.valid_size * num_train))
            split2 = int(np.floor(self.test_size * num_train))
            valid_idx, test_idx, train_idx = indices[:split], indices[split:split+split2], indices[split+split2:]
        elif self.splitting == 'scaffold':
            train_idx, valid_idx, test_idx = scaffold_split(train_dataset, self.valid_size, self.test_size)
        elif self.splitting == 'external':
            # MT patch: use pre-computed split indices (from clean_pipeline_v1/splits/...)
            train_idx, valid_idx, test_idx = self.external_split

        # MT patch: train uses random sampler (training noise OK).
        # Val + test MUST be SEQUENTIAL — otherwise pred order varies across
        # runs/ensemble members and ensembling pred matrices becomes meaningless
        # (each row index references a different molecule per member).
        from torch.utils.data import SubsetRandomSampler as _SRS
        class SubsetSequentialSampler:
            def __init__(self, indices): self.indices = list(indices)
            def __iter__(self): return iter(self.indices)
            def __len__(self): return len(self.indices)
        train_sampler = _SRS(train_idx)
        valid_sampler = SubsetSequentialSampler(valid_idx)
        test_sampler  = SubsetSequentialSampler(test_idx)

        # Build the graph cache here, in the parent, so forked dataloader workers inherit
        # it copy-on-write instead of each rebuilding it every epoch. Restricted to the
        # split indices so an index the loaders never touch can't newly raise here.
        train_dataset.precompute(list(train_idx) + list(valid_idx) + list(test_idx))

        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, sampler=train_sampler,
            num_workers=self.num_workers, drop_last=False
        )
        valid_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, sampler=valid_sampler,
            num_workers=self.num_workers, drop_last=False
        )
        test_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, sampler=test_sampler,
            num_workers=self.num_workers, drop_last=False
        )

        return train_loader, valid_loader, test_loader
