import torch
from torch.nn import functional as F

from torchdrug import tasks, core, metrics, layers
from torchdrug.core import Registry as R

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
     
    def preprocess(self, train_set, valid_set, test_set):
        hidden_dims = [self.model.output_dim] * (self.num_mlp_layer - 1)
        self.mlp = layers.MLP(self.model.output_dim, hidden_dims + [sum(self.num_class)])

    def forward(self, batch):
        all_loss = torch.tensor(0, dtype=torch.float32, device=self.device)
        metric = {}
        
        pred = self.predict(batch, all_loss, metric)  # predict label (BS,2065)
        target = self.target(batch)  # groundtruth label  (2065,1)
        metric.update(self.evaluate(pred, target))
        
        loss = F.cross_entropy(pred, torch.squeeze(target))
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
        batch_label = batch["superfamily_label"].unsqueeze(1)
        return batch_label

    def evaluate(self, pred, target):
        roc = metrics.area_under_roc(torch.argmax(pred), target)
        pr = metrics.area_under_prc(torch.argmax(pred), target)        
        return {
            "AUROC": roc,
            "AUPR": pr,
        }