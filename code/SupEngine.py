import os
import sys
import logging
from itertools import islice

import torch
from torch import distributed as dist
from torch import nn
from torch.utils import data as torch_data

from torchdrug import data, core, utils
from torchdrug.core import Registry as R
from torchdrug.utils import comm, pretty

module = sys.modules[__name__]
logger = logging.getLogger(__name__)

'''
    Customize our own core.Engine
    The mainly modification is the evaluate func
    - We will add our own metric: TMScore
    - And we can get the predict-label in training and evaluate steps
'''
class SupEngine(core.Engine):
    
    def __init__(self, task, train_set, valid_set, test_set, optimizer, scheduler=None, gpus=None, batch_size=1,
                 gradient_interval=1, num_worker=0, logger="logging", log_interval=100):
        self.rank = comm.get_rank()
        self.world_size = comm.get_world_size()
        self.gpus = gpus
        self.batch_size = batch_size
        self.gradient_interval = gradient_interval
        self.num_worker = num_worker
        
        if gpus is None:
                self.device = torch.device("cpu")
        else:
            if len(gpus) != self.world_size:
                error_msg = "World size is %d but found %d GPUs in the argument"
                if self.world_size == 1:
                    error_msg += ". Did you launch with `python -m torch.distributed.launch`?"
                raise ValueError(error_msg % (self.world_size, len(gpus)))
            self.device = torch.device(gpus[self.rank % len(gpus)])

        if self.world_size > 1 and not dist.is_initialized():
            if self.rank == 0:
                module.logger.info("Initializing distributed process group")
            backend = "gloo" if gpus is None else "nccl"
            comm.init_process_group(backend, init_method="env://")

        if hasattr(task, "preprocess"):
            if self.rank == 0:
                module.logger.warning("Preprocess training set")
            # TODO: more elegant implementation
            # handle dynamic parameters in optimizer
            old_params = list(task.parameters())
            result = task.preprocess(train_set, valid_set, test_set)
            if result is not None:
                train_set, valid_set, test_set = result
            new_params = list(task.parameters())
            if len(new_params) != len(old_params):
                optimizer.add_param_group({"params": new_params[len(old_params):]})
        if self.world_size > 1:
            task = nn.SyncBatchNorm.convert_sync_batchnorm(task)
        if self.device.type == "cuda":
            task = task.cuda(self.device)

        self.model = task
        self.train_set = train_set
        self.valid_set = valid_set
        self.test_set = test_set
        self.optimizer = optimizer
        self.scheduler = scheduler

        if isinstance(logger, str):
            if logger == "logging":
                logger = core.LoggingLogger()
            elif logger == "wandb":
                logger = core.WandbLogger(project=task.__class__.__name__)
            else:
                raise ValueError("Unknown logger `%s`" % logger)
        self.meter = core.Meter(log_interval=log_interval, silent=self.rank > 0, logger=logger)
        self.meter.log_config(self.config_dict())
    
    def train(self, num_epoch=1, batch_per_epoch=None):
        sampler = torch_data.DistributedSampler(self.train_set, self.world_size, self.rank)
        dataloader = data.DataLoader(self.train_set, self.batch_size, sampler=sampler, num_workers=self.num_worker)        
        batch_per_epoch = len(dataloader)
        for i_epoch in self.meter(num_epoch):
            metrics = []
            start_id = 0
            # the last gradient update may contain less than gradient_interval batches
            gradient_interval = min(batch_per_epoch - start_id, self.gradient_interval)
            
            for batch_id, batch in enumerate(islice(dataloader, batch_per_epoch)):
                if self.device.type == "cuda":
                    batch = utils.cuda(batch, device=self.device)

                loss, metric = self.model(batch)
                if not loss.requires_grad:
                    raise RuntimeError("Loss doesn't require grad. Did you define any loss in the task?")
                loss.backward()
                metrics.append(metric)
                
                if batch_id - start_id + 1 == gradient_interval:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    metric = utils.stack(metrics, dim=0)
                    metric = utils.mean(metric, dim=0)
                    if self.world_size > 1:
                        metric = comm.reduce(metric, op="mean")
                    
                    self.meter.update(metric)
                    metrics = []
                    start_id = batch_id + 1
                    gradient_interval = min(batch_per_epoch - start_id, self.gradient_interval)

            if self.scheduler:
                self.scheduler.step()
        

    @torch.no_grad()
    def evaluate(self, split, log=True):
        print(f'\n\n Run on {split}...')
        test_set = getattr(self, "%s_set" % split)
        sampler = torch_data.DistributedSampler(test_set, self.world_size, self.rank)
        dataloader = data.DataLoader(test_set, self.batch_size, sampler=sampler, num_workers=self.num_worker)    
        self.model.eval()
        preds = []
        targets = []
        csv_idxes = []
        for batch in dataloader:
            if self.device.type == "cuda":
                batch = utils.cuda(batch, device=self.device)
          
            pred = self.model.predict(batch)
            target, csv_idx = self.model.target(batch)
            preds.append(pred)
            targets.append(target)
            csv_idxes.append(csv_idx)
            
        pred = utils.cat(preds)
        target = utils.cat(targets)
        csv_idx = utils.cat(csv_idxes)
        if self.world_size > 1:
            pred = comm.cat(pred)
            target = comm.cat(target)
        metric = self.model.evaluate_tm(pred, target, csv_idx)
        self.meter.log(metric, category="%s/epoch" % split)

        return metric    