import logging
import os
import sys

import torch
from datetime import datetime
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils import save_config_file, accuracy, save_checkpoint
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(0)


class LinearProbing(object):

    def __init__(self, *args, **kwargs):
        self.args = kwargs['args']
        self.model = kwargs['model'].to(self.args.device)
        self.optimizer = kwargs['optimizer']
        self.scheduler = kwargs['scheduler']
        self.writer = SummaryWriter(kwargs['run_name'])
        logging.basicConfig(filename=os.path.join(self.writer.log_dir, 'training.log'), level=logging.DEBUG)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.args.device)

    def train(self, train_loader):

        scaler = GradScaler('cuda',enabled=self.args.fp16_precision)

        # save config file
        save_config_file(self.writer.log_dir, self.args)

        n_iter = 0
        logging.info(f"Start Linear_Probing training for {self.args.epochs} epochs.")
        logging.info(f"Training with gpu: {self.args.disable_cuda}.")

        self.model.backbone.eval()
        self.model.backbone.fc.train()
        
        for epoch_counter in range(self.args.epochs):
            
            for images, labels in tqdm(train_loader):
                
                images = images.to(self.args.device)
                labels = labels.to(self.args.device)

                with autocast('cuda', enabled=self.args.fp16_precision):
                    logits = self.model(images)
                    loss = self.criterion(logits, labels)

                self.optimizer.zero_grad()

                scaler.scale(loss).backward()

                scaler.step(self.optimizer)
                scaler.update()

                if n_iter % self.args.log_every_n_steps == 0:
                    top1, top5 = accuracy(logits, labels, topk=(1, 5))
                    self.writer.add_scalar('loss', loss, global_step=n_iter)
                    self.writer.add_scalar('acc/top1', top1[0], global_step=n_iter)
                    self.writer.add_scalar('acc/top5', top5[0], global_step=n_iter)
                    self.writer.add_scalar('learning_rate', self.scheduler.get_lr()[0], global_step=n_iter)

                n_iter += 1

            # warmup for the first 10 epochs
            if epoch_counter >= 10:
                self.scheduler.step()
            logging.debug(f"Epoch: {epoch_counter}\tLoss: {loss}\tTop1 accuracy: {top1[0]}")
            if epoch_counter % 30 == 0:
                # save model checkpoints in a folder
                mkdir_path = os.path.join(self.writer.log_dir, 'checkpointsLP')
                os.makedirs(mkdir_path, exist_ok=True)
                checkpoint_name = (
                    f'checkpointLP_{datetime.now().strftime("%Y%m%d_%H%M%S")}_{epoch_counter:04d}.pth.tar')
                save_checkpoint({
                    'epoch': epoch_counter,
                    'arch': self.args.arch,
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                }, is_best=False, filename=os.path.join(mkdir_path, checkpoint_name))


        logging.info("Training has finished.")
        # save model checkpoints
        checkpoint_name = (
                    f'checkpointLP_{datetime.now().strftime("%Y%m%d_%H%M%S")}_{epoch_counter:04d}.pth.tar')
        save_checkpoint({
            'epoch': self.args.epochs,
            'arch': self.args.arch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, is_best=False, filename=os.path.join(mkdir_path, checkpoint_name))
        logging.info(f"Model checkpoint and metadata has been saved at {self.writer.log_dir}.")
        

    def test(self, test_loader):
        self.model.eval()
        total = 0
        correct = 0

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels in tqdm(test_loader):
                images = images.to(self.args.device)
                labels = labels.to(self.args.device)

                logits = self.model(images)
                _, predicted = torch.max(logits, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                all_preds.append(predicted.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        acc = 100 * correct / total

        # concatenate batches
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)

        # confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
       # plot confusion matrix
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        tick_marks = np.arange(len(set(all_labels)))
        plt.xticks(tick_marks, tick_marks)
        plt.yticks(tick_marks, tick_marks)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.savefig(os.path.join(self.writer.log_dir, 'confusion_matrix.png'))  

        print(f'Test Accuracy of the model on the test images: {acc:.2f} %')
        print('Confusion Matrix:')
        print(cm)

        logging.info(f'Test Accuracy of the model on the test images: {acc:.2f} %')
        logging.info(f'Confusion Matrix:\n{cm}')

        return acc, cm
