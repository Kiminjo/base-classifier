from pathlib import Path 
from PIL import Image 
import random

from torch.utils.data import Dataset
from torchvision import transforms 

random.seed(21)

class DogCat(Dataset):
    def __init__(self, 
                src: str,
                transform: transforms):
        imgs = [str(p) for p in Path(src).glob("*/*.jpg")]
        self.imgs = random.sample(imgs, 200)
        self.labels = [1 if 'dog' == Path(p).parent.name else 0 for p in self.imgs]
        self.transform = transform

    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        img = Image.open(self.imgs[idx])
        label = self.labels[idx]

        if self.transform:
            img = self.transform(img)
        return img, label



