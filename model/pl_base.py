from torchmetrics import Accuracy
from torch.nn import functional as F
from torch.utils.data import DataLoader
from utils import Grid
import torch

BATCH_SIZE = 256 if torch.cuda.is_available() else 64

class Model(LightningModule):
    def __init__(self, model, data_dir, size=(3, 3), hidden_size=64, learning_rate=2e-4):
        super().__init__()

        # Set our init args as class attributes
        self.grid_size = size
        self.data_dir = data_dir
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate

        # Hardcode some dataset specific attributes

        self.model = model
        self.transform = None

        self.val_accuracy = Accuracy()
        self.test_accuracy = Accuracy()

    def forward(self, x):
        x = self.model(x)
        return F.log_softmax(x, dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.val_accuracy.update(preds, y)

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.val_accuracy, prog_bar=True)

    # def test_step(self, batch, batch_idx):
    #     x, y = batch
    #     logits = self(x)
    #     loss = F.nll_loss(logits, y)
    #     preds = torch.argmax(logits, dim=1)
    #     self.test_accuracy.update(preds, y)

    #     # Calling self.log will surface up scalars for you in TensorBoard
    #     self.log("test_loss", loss, prog_bar=True)
    #     self.log("test_acc", self.test_accuracy, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    ####################
    # DATA RELATED HOOKS
    ####################

    def prepare_data(self):
        # download
        Grid(self.data_dir, size=self.size, download=True, strip=False)
        Grid(self.data_dir, size=self.size, download=True, strip=True)

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            ds = Grid(self.data_dir, size=self.size, transform=self.transform)
            train_size = int(0.8 * len(ds))
            test_size = len(ds) - train_size
            self.train_ds, self.val_ds = torch.utils.data.random_split(ds, [train_size, test_size])

        # if stage == "test" or stage is None:
        #     self.test_ds = Grid(self.data_dir, size=self.size, train=False, transform=self.transform)


    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=BATCH_SIZE, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=BATCH_SIZE)

    # def test_dataloader(self):
    #     return DataLoader(self.mnist_test, batch_size=BATCH_SIZE)