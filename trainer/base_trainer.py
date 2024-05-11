import os
import numpy as np
import warnings
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
from tensorboardX import SummaryWriter
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from model.pipeline import build_pipeline
from model.criterions import build_criterions
from input_pipeline import build_dataloader
from evaluators import build_evaluator
from utils.registry import register_module

warnings.filterwarnings("ignore")

@register_module(parent="trainers")
def base_trainer(cfg):
    return BaseTrainer(cfg)

class BaseTrainer():
    def __init__(self, cfg):
        self.cfg = cfg
        current_time = datetime.now()
        formatted_date = current_time.strftime("%Y.%m.%d")
        formatted_datetime = formatted_date + " " + current_time.strftime("%H:%M")
        self.workspace = os.path.join(self.cfg['general']['workspace'], formatted_datetime)
        if not os.path.exists(self.workspace):
            os.makedirs(self.workspace)

        self.max_epoch = self.cfg['general']['epoch']
        self.evaluator = build_evaluator(self.cfg['evaluator']['type'], self.cfg)
        self.loss_func = build_criterions(cfg)
        self.net = build_pipeline(self.cfg)

        print(self.net)

        if torch.cuda.is_available():
            self.net = self.net.cuda()
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        print(f"use device {self.device}")

        # Init optimizer
        params = [
            {"params": self.net.parameters(), "lr": self.cfg['train']["initial_lr"]}
            ]
        optim_params = self.cfg["optim.params"]
        self.optimizer = eval("optim.{}".format(self.cfg["train"]['optimizer']))(params, **optim_params)

        if self.cfg["train"]["lr_strategy"] == "Cosine":
            self.scheduler = lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100)
        else:
            self.scheduler = None
        # load dataloader
        self.train_loader = build_dataloader(self.cfg, 'train')
        self.val_loader = build_dataloader(self.cfg, 'val')
        #log visual
        self.writer = SummaryWriter(self.workspace)

    def train_epoch(self, epoch):
        self.net.train()
        train_map=[]
        for i, data in enumerate(tqdm(self.train_loader, desc=f"train {epoch}"), 0):  
            # get the inputs
            inputs, labels = data
            # inputs&labels load in Variable
            inputs, labels = Variable(inputs).to(self.device), Variable(labels).to(self.device)#batch,22,1000
            self.optimizer.zero_grad()
            output, _ =self.net(inputs)
            loss = self.loss_func(output, labels.long())
            loss.backward()  
            self.optimizer.step()
            pred = output.argmax(dim=1, keepdim=True)
            acc = self.evaluator.accuracy(labels.cpu(), pred.cpu())
            train_map.append(acc)
            self.writer.add_scalar('train/Loss', loss, i)
            self.writer.add_scalar('train/mAP', acc, i)

        if self.scheduler is not None:
            self.scheduler.step()   
            lr_curr = self.scheduler.get_lr()[0]
            self.writer.add_scalar('train/LR', lr_curr, epoch)

        print("Train Epoch: {}\ttrain_Loss: {:.6f} mAP:{:.4f}".format(
            epoch, 
            loss.item(), 
            np.mean(train_map)
        ))
            
    def val(self, epoch):
        self.net.eval()
        test_loss = 0
        test_map = []
        with torch.no_grad():
            for i, data in enumerate(tqdm(self.val_loader, desc=f"val {epoch}"), 0):
                inputs, labels = data
                inputs, labels = Variable(inputs).to(self.device), Variable(labels).to(self.device)
                labels=labels.long()
                output, _ = self.net(inputs)
                
                test_loss += self.loss_func(output, labels.long())
                pred = output.argmax(dim=1, keepdim=True)
                acc = self.evaluator.accuracy(labels.cpu(), pred.cpu())
                test_map.append(acc)
        test_loss /= len(self.val_loader.dataset)
        
        self.writer.add_scalar('Val/Loss', test_loss, epoch)
        self.writer.add_scalar('Val/mAP', np.round(np.mean(test_map), 4), epoch)

        print('Val set: Average loss: {:.4f}, mAP: {:.4f}'.format(
                test_loss,
                np.mean(test_map)
            ))
        return np.round(np.mean(test_map), 4)
    
    def train(self, ):
        print("start traing...")
        for epoch in range(1, self.max_epoch + 1):
            self.train_epoch(epoch)
            val_map = self.val(epoch)
            if os.path.exists(self.cfg['general']['workspace'])==False:
                os.makedirs(self.cfg['general']['workspace'])
            save_epoch_path = os.path.join(self.workspace, str(epoch)+'-'+str(val_map)+'.pt')
            torch.save(self.net.state_dict(),
                        save_epoch_path)
            latest_path = os.path.join(self.workspace, 'latest.pt')
            torch.save(self.net.state_dict(), latest_path)