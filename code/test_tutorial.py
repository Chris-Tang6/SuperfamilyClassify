import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import time
import pandas as pd
from tqdm import tqdm
import torch
from torch.utils import data as torch_data
from torchdrug import data, datasets, models, tasks, core, transforms
from torchdrug import layers
from torchdrug.layers import geometry


class EnzymeCommissionToy(datasets.EnzymeCommission):
    url = "https://miladeepgraphlearningproteindata.s3.us-east-2.amazonaws.com/data/EnzymeCommission.tar.gz"
    md5 = "728e0625d1eb513fa9b7626e4d3bcf4d"
    processed_file = "enzyme_commission_toy.pkl.gz"
    test_cutoffs = [0.3, 0.4, 0.5, 0.7, 0.95]


truncate_transform = transforms.TruncateProtein(max_length=350, random=False)
protein_view_transform = transforms.ProteinView(view="residue")
transform = transforms.Compose([truncate_transform, protein_view_transform])

start_time = time.time()
dataset = EnzymeCommissionToy("~/protein-datasets/", transform=transform, atom_feature=None,
                              bond_feature=None)
end_time = time.time()
print("Duration of second instantiation: ", end_time - start_time)

train_set, valid_set, test_set = dataset.split()
print("Shape of function labels for a protein: ", dataset[0]["targets"].shape)
print("train samples: %d, valid samples: %d, test samples: %d" % (len(train_set), len(valid_set), len(test_set)))

graph_construction_model = layers.GraphConstruction(node_layers=[geometry.AlphaCarbonNode()],
                                                    edge_layers=[geometry.SpatialEdge(radius=10.0, min_distance=5),
                                                                 geometry.KNNEdge(k=10, min_distance=5),
                                                                 geometry.SequentialEdge(max_distance=2)])
gearnet = models.GearNet(input_dim=21,
                         hidden_dims=[512, 512, 512],
                         num_relation=7,
                         batch_norm=True,
                         concat_hidden=True,
                         short_cut=True,
                         readout='sum')

gear_edge = models.GearNet(input_dim=21,
                           hidden_dims=[512, 512, 512],
                           num_relation=7,
                           edge_input_dim=59,
                           num_angle_bin=8,
                           batch_norm=True,
                           concat_hidden=True,
                           short_cut=True,
                           readout='sum')

graph_construction_model = layers.GraphConstruction(node_layers=[geometry.AlphaCarbonNode()],
                                                   edge_layers=[geometry.SpatialEdge(radius=10.0, min_distance=5),
                                                               geometry.KNNEdge(k=10, min_distance=5),
                                                               geometry.SequentialEdge(max_distance=2)],
                                                   edge_feature="gearnet")
task = tasks.MultipleBinaryClassification(gearnet, graph_construction_model=graph_construction_model,
                                         num_mlp_layer=3, task=[_ for _ in range(len(dataset.tasks))],
                                          criterion="bce", metric=["auprc@micro", "f1_max"])

optimizer = torch.optim.Adam(task.parameters(), lr=1e-4)
solver = core.Engine(task, train_set, valid_set, test_set, optimizer,
                     gpus=[1], batch_size=4)
solver.train(num_epoch=10)
solver.evaluate("valid")