import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
import time
import pandas as pd
from tqdm import tqdm
import torch
from torch.utils import data as torch_data
from torchdrug import data, models, tasks, core, transforms
from torchdrug import layers
from torchdrug.layers import geometry


class ScopeSuperFamilyClassify(data.ProteinDataset):
    source_file = "pdb_scope_db40.csv"
    pdb_dir = "dbstyle_all-40-2.08"
    processed_file = 'scope_superfamily.pkl.gz'

    label2id_dir = 'pdb_scope_label2id.pkl'
    id2label_dir = 'pdb_scope_id2label.pkl'

    splits = ["train", "valid", "test"]
    split_ratio = [0.8, 0.1]
    target_fields = ["superfamily_label"]  # label column

    def __init__(self, path='/home/tangwuguo/datasets/scope40', verbose=1, **kwargs):
        if not os.path.exists(path):
            raise FileExistsError("Unknown path `%s` for SCOPE dataset" % path)
        self.path = path
        df = pd.read_csv(os.path.join(path, self.source_file))
        pkl_file = os.path.join(path, self.processed_file)

        if os.path.exists(pkl_file):
            # load processed pkl
            self.load_pickle(pkl_file, verbose=verbose, **kwargs)
        else:
            pdb_files = df['id']
            pdb_files = pdb_files.apply(lambda x: os.path.join(path, self.pdb_dir, x + '.ent')).tolist()
            self.load_pdbs(pdb_files=pdb_files, verbose=1)
            self.save_pickle(pkl_file, verbose=verbose)

        len = df['id'].size
        train_size = int(len * self.split_ratio[0])
        valid_size = int(len * self.split_ratio[1])
        test_size = len - train_size - valid_size
        self.num_samples = [train_size, valid_size, test_size]
        self.targets = {'superfamily_label': torch.tensor(df['label'].tolist())}

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
        if self.lazy:
            protein = self.load_hdf5(self.pdb_files[index])
        else:
            protein = self.data[index].clone()
        if hasattr(protein, "residue_feature"):
            with protein.residue():
                protein.residue_feature = protein.residue_feature.to_dense()
        item = {"graph": protein, "superfamily_label": self.targets["superfamily_label"][index]}
        if self.transform:
            item = self.transform(item)
        return item


truncate_transform = transforms.TruncateProtein(max_length=350, random=False)
protein_view_transform = transforms.ProteinView(view="residue")
transform = transforms.Compose([truncate_transform, protein_view_transform])

start_time = time.time()
dataset = ScopeSuperFamilyClassify("/home/tangwuguo/datasets/scope40", transform=transform)
end_time = time.time()
print("Duration of second instantiation: ", end_time - start_time)


train_set, valid_set, test_set = dataset.split()
print("Shape of function labels for a protein: ", dataset[0]["superfamily_label"].shape)
print("train samples: %d, valid samples: %d, test samples: %d" % (len(train_set), len(valid_set), len(test_set)))

graph_construction_model = layers.GraphConstruction(node_layers=[geometry.AlphaCarbonNode()],
                                                    edge_layers=[geometry.SpatialEdge(radius=10.0, min_distance=5),
                                                                 geometry.KNNEdge(k=10, min_distance=5),
                                                                 geometry.SequentialEdge(max_distance=2)])
gearnet = models.GearNet(input_dim=21, hidden_dims=[512, 512, 512], num_relation=7,
                         batch_norm=True, concat_hidden=True, short_cut=True, readout="sum")

gearnet_edge = models.GearNet(input_dim=21, hidden_dims=[512, 512, 512],
                              num_relation=7, edge_input_dim=59, num_angle_bin=8,
                              batch_norm=True, concat_hidden=True, short_cut=True, readout="sum")

graph_construction_model = layers.GraphConstruction(node_layers=[geometry.AlphaCarbonNode()],
                                                    edge_layers=[geometry.SpatialEdge(radius=10.0, min_distance=5),
                                                                 geometry.KNNEdge(k=10, min_distance=5),
                                                                 geometry.SequentialEdge(max_distance=2)],
                                                    edge_feature="gearnet")

task = tasks.PropertyPrediction(gearnet_edge, graph_construction_model=graph_construction_model, num_mlp_layer=3,
                                task=dataset.tasks, num_class=2065, criterion="ce", metric=["auprc", "auroc"], verbose=1)

optimizer = torch.optim.Adam(task.parameters(), lr=1e-4)
solver = core.Engine(task, train_set, valid_set, test_set, optimizer,
                     gpus=[1], batch_size=32)
solver.train(num_epoch=1)
solver.evaluate("valid")
# solver.evaluate("test")