import torch
from torch.nn import functional as F

from torchdrug import tasks, core, metrics, layers
from torchdrug.core import Registry as R

from TMScore import getTMScore
import pandas as pd
import os
from sklearn.metrics import accuracy_score, f1_score

@R.register("tasks.SuperfamilyPrediction")
class SuperfamilyPrediction(tasks.Task, core.Configurable):
    """ 
        SuperfamilyPrediction is a multilabel classification task.        
    """ 
    def __init__(self, model, num_mlp_layer=1, num_class=None, 
                 graph_construction_model=None, verbose=0):
        super(SuperfamilyPrediction, self).__init__()
        self.model = model
        self.num_mlp_layer = num_mlp_layer
        self.num_class = (num_class,) if isinstance(num_class, int) else num_class
        self.graph_construction_model = graph_construction_model
        self.verbose = verbose
        
        self.path = '/home/tangwuguo/datasets/scope40_s'
        self.dataset_file = 'scop40_s.csv'
        self.pdb_dir = 'ents'
        self.df = pd.read_csv(os.path.join(self.path, self.dataset_file))
     
    def preprocess(self, train_set, valid_set, test_set):
        hidden_dims = [self.model.output_dim] * (self.num_mlp_layer - 1)
        self.mlp = layers.MLP(self.model.output_dim, hidden_dims + [sum(self.num_class)])

    def forward(self, batch):
        all_loss = torch.tensor(0, dtype=torch.float32, device=self.device)
        metric = {}
        
        pred = self.predict(batch, all_loss, metric)  # predict label (BS,2065)
        target, csv_idx = self.target(batch)  # groundtruth label  (BS,1)
        metric.update(self.evaluate(pred, target, csv_idx))
        
        loss = F.cross_entropy(pred, target)
        metric["ce loss"] = loss
        all_loss += loss
        return all_loss, metric
     
    def predict(self, batch, all_loss=None, metric=None):
        graph = batch['graph']
        if self.graph_construction_model:
            graph = self.graph_construction_model(graph)
        output = self.model(graph, graph.node_feature.float(),
                            all_loss=all_loss, metric=metric)        
        pred = self.mlp(output["graph_feature"])
        return pred

    def target(self, batch):
        batch_label = batch["superfamily_label"]
        batch_csv_idx = batch["csv_idx"]
        return batch_label, batch_csv_idx  # (n,)

    def evaluate(self, pred, target, csv_idx):
        pred_label = torch.argmax(pred, dim=1)  # (n,)
        # roc = metrics.area_under_roc(pred_label, target)
        # pr = metrics.area_under_prc(pred_label, target)
        acc = accuracy_score(pred_label.tolist(), target.tolist())        
        f1 = f1_score(pred_label.tolist(), target.tolist(), average='micro')        
                     
        return {
            "ACC": torch.tensor(acc),            
            "F1": torch.tensor(f1),            
        }
    
    def evaluate_tm(self, pred, target, csv_idx):        
        pred_label = torch.argmax(pred, dim=1)  # (n,)
        acc = accuracy_score(pred_label.tolist(), target.tolist())        
        f1 = f1_score(pred_label.tolist(), target.tolist(), average='macro')
        
        pdb_names = self.df.iloc[csv_idx.tolist()]['id'].tolist()
        pdb_names = [os.path.join(self.path, self.pdb_dir, i + '.ent') for i in pdb_names]                    
        tm_list, tm_score = getTMScore(pdb_names, target)  # TODO by watch tm_list to find weak pred samples               
        return {
            "ACC": torch.tensor(acc),            
            "F1": torch.tensor(f1),
            "TMScore": torch.tensor(tm_score),
        }