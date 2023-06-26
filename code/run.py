from torchdrug import core, data, transforms, models, layers, utils
from torchdrug.layers import geometry
from torchdrug.utils import comm
import os
import torch
from torch import distributed as dist
from torch import nn
from torch.utils import data as torch_data
from itertools import islice

from ScopeSuperFamilyClassify import ScopeSuperFamilyClassify
from SuperFamilyTask import SuperfamilyPrediction
from SupEngine import SupEngine
import yaml


with open("./config/train.yaml", "r") as f:
    params = yaml.safe_load(f)

truncate_transform = transforms.TruncateProtein(max_length=350, random=False)
protein_view_transform = transforms.ProteinView(view="residue")
transform = transforms.Compose([truncate_transform, protein_view_transform])

dataset = ScopeSuperFamilyClassify(params['dataset'], transform=transform)
train_set, valid_set, test_set = dataset.split()
print("train samples: %d, valid samples: %d, test samples: %d" % (len(train_set), len(valid_set), len(test_set)))

# GearNet model
params_m = params['task']['model']
gearnet = models.GearNet(input_dim=params_m['input_dim'], 
                         hidden_dims=params_m['hidden_dims'], 
                         num_relation=params_m['num_relation'],
                         batch_norm=params_m['batch_norm'],
                         concat_hidden=params_m['concat_hidden'],
                         short_cut=params_m['short_cut'],
                         readout=params_m['readout'])

graph_construction_model = layers.GraphConstruction(node_layers=[geometry.AlphaCarbonNode()],
                                                    edge_layers=[geometry.SpatialEdge(radius=10.0, min_distance=5),
                                                                 geometry.KNNEdge(k=10, min_distance=5),
                                                                 geometry.SequentialEdge(max_distance=2)],
                                                    edge_feature="gearnet")

# Superfamily classify task
task = SuperfamilyPrediction(gearnet, params=params, 
                             graph_construction_model=graph_construction_model, verbose=1)

optimizer = torch.optim.Adam(task.parameters(), lr=float(params['engine']['lr']))
solver = SupEngine(task, train_set, valid_set, test_set, optimizer,
                     gpus=params['engine']['gpus'], 
                     batch_size=params['engine']['batch_size'])
solver.train(num_epoch=params['engine']['num_epoch'])
solver.evaluate("valid")
solver.evaluate("test")
