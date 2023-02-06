import sys 
sys.path.append("/home/injo/research/base-classifier")

import torch
import timm
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchmetrics.functional import accuracy, f1_score, precision, recall

import time

from data.cat_dog import DogCat

def evaluate(model,
            criterion,
            device, 
            loader
            ):
    labels, predictions = torch.tensor([]), torch.tensor([])
    losses = 0
    check_time = 0
    model.eval()

    with torch.no_grad():
        for img, label in loader:
            img = img.to(device)
            label = label.to(device)

            start = time.time()
            logit = model(img)
            end = time.time()
            loss = criterion(logit, label)
            pred = torch.argmax(logit, dim=1)
            
            losses += loss.item()
            labels = torch.cat((labels, label.detach().cpu())).to(torch.int8)
            predictions = torch.cat((predictions, pred.detach().cpu())).to(torch.int8)
            check_time += (end-start)

        losses /= len(loader.dataset)
        check_time /= len(loader.dataset)
    return labels, predictions, losses, check_time


if __name__=='__main__':
    model_name = "resnet50"
    device = "cuda:2"
    src = "/home/injo/solutions/luna/data/cat_dog"
    num_classes = 10

    # model init 
    model = timm.create_model(model_name,
                            pretrained=True,
                            num_classes=num_classes)
    criterion = nn.CrossEntropyLoss()
    model.to(device)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((360, 360))
    ])

    # data init 
    testset = DogCat(src=src,
                    transform=transform)
    test_loader = DataLoader(testset,
                            batch_size=1,
                            num_workers=8,
                            shuffle=False)
    
    # evaluate
    labels, predictions, loss, check_time = evaluate(model=model,
                                        criterion=criterion,
                                        device=device,
                                        loader=test_loader)

    metrics = {}
    metrics["accuracy"] = accuracy(predictions, labels).item()
    metrics["f1_score"] = f1_score(predictions, labels).item()
    metrics["precision"] = precision(predictions, labels).item()
    metrics["recall"] = recall(predictions, labels).item()
    metrics['time'] = check_time

    print(metrics)
    print("here")
    
    



        