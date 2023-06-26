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
import os
import time


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
        self.time = time.strftime("%Y-%m-%d_%H:%M",time.localtime())
        
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
        epoch_metrics = []
        best_f1 = -1
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
            
            # add epoch metrics to list
            m_acc = round(sum(self.meter.records['ACC'][-batch_per_epoch:])/batch_per_epoch, 4)
            m_f1 = round(sum(self.meter.records['F1'][-batch_per_epoch:])/batch_per_epoch, 4)
            m_ce = round(sum(self.meter.records['CELoss'][-batch_per_epoch:])/batch_per_epoch, 4)
            epoch_metrics.append((i_epoch, m_acc, m_f1, m_ce))
            # dump model
            if m_f1 > best_f1:
                best_f1 = m_f1           
                torch.save(self.model.state_dict(), f"./checkpoint/model_param{self.time}.pkl")
                module.logger.info(f"Save Model Params at Epoch-{i_epoch}") 
            if self.scheduler:
                self.scheduler.step()
        # write to log file...
        self.write_log(epoch_metrics)
        

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
    
    def write_log(self, log_list):
        log_dir_name = os.path.join('log', 'train_'+self.time+'.log')
        with open(log_dir_name, 'w+') as f:
            f.write(f'Epoch\tACC\tF1\tCELoss\n')
            for item in log_list:
                f.write(f'{item[0]}\t{item[1]}\t{item[2]}\t{item[3]}\n')
        module.logger.info(f"Training Log Dump Over at {log_dir_name}")           
            
        