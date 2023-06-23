import os 
import pandas as pd
from torchdrug import data
from torchdrug.core import Registry as R
import torch
from torch.utils import data as torch_data
from rdkit import Chem

from tqdm import tqdm
import logging
import warnings
logger = logging.getLogger(__name__)


@R.register("datasets.ScopeSuperFamilyClassify")
class ScopeSuperFamilyClassify(data.ProteinDataset):
    dataset_file = "pdb_scope_db40.csv"
    pdb_dir = "dbstyle_all-40-2.08"
    processed_file = 'scope_superfamily.pkl.gz'

    splits = ["train", "valid", "test"]
    split_ratio = [0.8, 0.1]
    target_fields = ["superfamily_label"]  # label column

    def __init__(self, path='/home/tangwuguo/datasets/scope40', 
                 dataset_file='pdb_scope_db40.csv',  
                 pdb_dir = 'dbstyle_all-40-2.08', verbose=1, **kwargs):
        if not os.path.exists(path):
            raise FileExistsError("Unknown path `%s` for SCOPE dataset" % path)
        self.path = path
        self.dataset_file = dataset_file
        self.pdb_dir = pdb_dir
        
        self.df = pd.read_csv(os.path.join(path, self.dataset_file))
        pkl_file = os.path.join(path, self.processed_file)

        if os.path.exists(pkl_file):
            # load processed pkl
            self.load_pickle(pkl_file, verbose=verbose, **kwargs)
        else:
            pdb_files = self.df['id']
            pdb_files = pdb_files.apply(lambda x: os.path.join(path, self.pdb_dir, x + '.ent')).tolist()
            self.load_pdbs_drop(pdb_files=pdb_files, verbose=1, **kwargs)
            self.save_pickle(pkl_file, verbose=verbose)

        len = self.df['id'].size
        train_size = int(len * self.split_ratio[0])
        valid_size = int(len * self.split_ratio[1])
        test_size = len - train_size - valid_size
        self.num_samples = [train_size, valid_size, test_size]
        self.targets = {'superfamily_label': torch.tensor(self.df['label'].tolist())}

    def split(self, keys=None):
        keys = keys or self.splits
        offset = 0
        splits = []
        for split_name, num_sample in zip(self.splits, self.num_samples):
            if split_name in keys:
                split = torch_data.Subset(self, range(offset, offset + num_sample))
                splits.append(split)
            offset += num_sample
        return splits

    def get_item(self, index):
        protein = self.data[index].clone()
        if hasattr(protein, "residue_feature"):
            with protein.residue():
                protein.residue_feature = protein.residue_feature.to_dense()
        item = {"graph": protein, 
                "superfamily_label": self.targets["superfamily_label"][index], 
                "csv_idx": index}
        if self.transform:
            item = self.transform(item)
        return item
    
    def load_pdbs_drop(self, pdb_files, transform=None, verbose=0):
        self.data = []
        self.pdb_files = []
        self.sequences = []
        self.transform = transform

        drop_ids = []
        drop_pdbs = []
        if verbose:
            pdb_files = tqdm(pdb_files, "Constructing proteins from pdbs")
        for i, pdb_file in enumerate(pdb_files):
            mol = Chem.MolFromPDBFile(pdb_file)
            if not mol:
                logger.debug("Can't construct molecule from pdb file `%s`. Drop this sample." % pdb_file)
                drop_ids.append(i)
                drop_pdbs.append(pdb_file)
                continue
            protein = data.Protein.from_molecule(mol)
            if not protein:
                logger.debug("Can't construct protein from pdb file `%s`. Drop this sample." % pdb_file)
                drop_ids.append(i)
                drop_pdbs.append(pdb_file)
                continue

            if hasattr(protein, "residue_feature"):
                with protein.residue():
                    protein.residue_feature = protein.residue_feature.to_sparse()
            self.data.append(protein)
            self.pdb_files.append(pdb_file)
            self.sequences.append(protein.to_sequence() if protein else None)
        # after drop loop, need to reset the idx
        print(f'Drop ids {drop_ids}, they are {drop_pdbs}.')
        self.df = self.df.drop(self.df.index[drop_ids])
        self.df.reset_index(drop=True)
        self.df.to_csv(os.path.join(self.path, self.dataset_file), index=None)
        