import torch
import timm
from torch import nn, optim 
from torchmetrics.functional import accuracy, f1_score, precision, recall

class Trainer:
    def __init__(self):
        self.model_name = 'resnet50'
        self.epochs = 100
        self.num_classes = 10
        self.lr = 1e-4
        self.device = 'cuda:2'

    def train(self):
        self._before_train()
        for epoch in range(1, self.epochs+1):
            labels, predictions, epoch_loss = self._train_epoch()
            self._after_epoch(labels, predictions, epoch_loss, epoch)
        
                
    def _before_train(self): 
        # model initialize
        self.model = timm.create_model(self.model_name,
                                    pretrained=True,
                                    num_classes=self.num_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.paramters(),
                                    lr=self.lr)
    
    def _train_iter(self):
        gts, preds = torch.tensor([]), torch.tensor([])
        epoch_loss = 0
        self.model.train()

        for img, label in self.train_loader:
            img = img.to(self.device)
            label = label.to(self.device)

            logit = self.model(img)
            pred = torch.argmax(logit, dim=1)
            loss = self.criterion(label, logit)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            gts = torch.cat((gts, label.detach().cpu()))
            preds = torch.cat((preds, pred.detach().cpu()))
            epoch_loss += loss.item()

        epoch_loss /= len(self.train_loader.dataset)
        return gts, preds, epoch_loss
    
    def _after_epoch(self,
                    labels, 
                    predictions,
                    epoch_loss,
                    epoch
                    ):
        metrics = {}
        metrics["accuracy"] = accuracy(predictions, labels)
        metrics["f1_score"] = f1_score(predictions, labels, average="macro")
        metrics["precision"] = precision(predictions, labels, average="macro")
        metrics["recall"] = recall(predictions, recall, average="macro")
        metrics["loss"] = epoch_loss 
        metrics["epoch"] = epoch

        if epoch == 0:
            best_metrics = metrics 
        if best_metrics["loss"] > metrics:
            best_metrics = metrics 



        
        








    