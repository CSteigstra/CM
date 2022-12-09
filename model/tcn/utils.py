import torch
from torch.utils.data import Dataset
import pandas as pd


class zero_y():
    def __init__(self, size):
        self.y = torch.zeros(size)

    def __getitem__(self, _):
        return self.y


class Grid(Dataset):
    def __init__(self, dir, size=(3, 3), transform=None, target_transform=None, strip=False):
        self.path = f'{dir}/grid_{size[0]}_{size[1]}_{"strip" if strip else "no_strip"}'

        self.x = pd.read_csv(f'{self.path}.csv')
        if strip:
            self.y = pd.read_csv(f'{self.path}_y.csv')
        else:
            self.y = zero_y(size)

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = self.x.iloc[idx, 1:]
        y = self.y[idx]
        
        if self.transform:
            x = self.transform(x)
        if self.target_transform:
            y = self.target_transform(y)

        return x, y

def data_generator(root, size=(3, 3), batch_size=1, strip=False):
    ds = Grid(root, size=size, strip=strip)

    train_size = int(0.8 * len(ds))
    test_size = len(ds) - train_size

    train_ds, test_ds = torch.utils.data.random_split(ds, [train_size, test_size])

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size)

    return train_loader, test_loader