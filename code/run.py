from torchdrug import datasets, core, tasks, transforms, models, layers
from torchdrug.layers import geometry
import torch
import os
from ScopeSuperFamilyClassify import ScopeSuperFamilyClassify
from SuperFamilyTask import SuperfamilyPrediction

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

db_path = '/home/tangwuguo/datasets/scope40_s'
db_file = 'scop40_s.csv'  # 'pdb_scope_db40.csv'
pdb_dir = 'ents'  #'dbstyle_all-40-2.08'

truncate_transform = transforms.TruncateProtein(max_length=350, random=False)
protein_view_transform = transforms.ProteinView(view="residue")
transform = transforms.Compose([truncate_transform, protein_view_transform])

dataset = ScopeSuperFamilyClassify(path= db_path, 
                                    dataset_file=db_file,
                                    pdb_dir=pdb_dir,
                                    transform=transform, verbose=1)

train_set, valid_set, test_set = dataset.split()
print("Shape of function labels for a protein: ", dataset[0]["superfamily_label"].shape)
print("train samples: %d, valid samples: %d, test samples: %d" % (len(train_set), len(valid_set), len(test_set)))

# GearNet model
gearnet = models.GearNet(input_dim=21, hidden_dims=[512, 512, 512], num_relation=7,
                         batch_norm=True, concat_hidden=True, short_cut=True, readout="sum")

graph_construction_model = layers.GraphConstruction(node_layers=[geometry.AlphaCarbonNode()],
                                                    edge_layers=[geometry.SpatialEdge(radius=10.0, min_distance=5),
                                                                 geometry.KNNEdge(k=10, min_distance=5),
                                                                 geometry.SequentialEdge(max_distance=2)],
                                                    edge_feature="gearnet")

# Superfamily classify task
task = SuperfamilyPrediction(gearnet, num_mlp_layer=3, 
                            graph_construction_model=graph_construction_model, 
                            num_class=2065, verbose=1)

optimizer = torch.optim.Adam(task.parameters(), lr=1e-4)
solver = core.Engine(task, train_set, valid_set, test_set, optimizer,
                     gpus=[0], batch_size=32)

solver.train(num_epoch=5)
solver.evaluate("valid")
solver.evaluate("test")