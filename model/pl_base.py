from torchmetrics import Accuracy, MetricCollection
from torch.nn import functional as F
from torch.utils.data import DataLoader
from utils import GridIter
import torch
from pytorch_lightning.trainer.supporters import CombinedLoader
import pytorch_lightning as pl
import itertools
import torch_geometric
import numpy as np
# from pytorch_lightning.trainer.suppporters import CombinedLoader


BATCH_SIZE = 256 if torch.cuda.is_available() else 64
VAL_SPLIT = 0.8


class Model(pl.LightningModule):
    def __init__(self, model, data_dir, train_sz=[(4,4),(5,5),(6,6)], test_sz=[(7,7),(8,8)], learning_rate=2e-4, transform_x=None, batch_size=BATCH_SIZE):
        super().__init__()

        # Set our init args as class attributes
        self.data_dir = data_dir
        self.learning_rate = learning_rate
        self.train_sz, self.test_sz = train_sz, test_sz
        self.max_sz = max([max(sz) for sz in train_sz + test_sz])

        self.batch_size = 256 // 2**len(train_sz)
        # self.batch_size = 128 * 2**(-int(np.log(len(train_sz))))

        # Hardcode some dataset specific attributes
        # self.conv_model = 

        self.model = model
        self.transform_x = transform_x

        # self.train_acc = torch.nn.ModuleDict({sz: Accuracy('binary') for sz in train_sz})
        # self.val_acc = torch.nn.ModuleDict({sz: Accuracy('binary') for sz in train_sz})
        self.loss = torch.nn.BCEWithLogitsLoss()
        # self.test_accuracy = Accuracy('binary')

        # Store hyperparameters
        self.save_hyperparameters(ignore=["model"])	

    def forward(self, x, y):
        # print(x.shape, y.shape)
        if x.shape[-1] == 8 or torch.rand(1)[0] > 0.5:
            x = x.repeat(1, 1, 2, 2)
        # if torch.rand(1)[0] > 0.5:

        x = self.model(x)
        # print(x.shape)
        x = x.squeeze()
        # y = torch.ones(x.shape).to(device=torch.device("cuda"))

        # print(F.sigmoid(x) >= 0.5)
        # print()
        # print(x.shape, x.dtype, y.shape, y.dtype)
        loss = self.loss(x, y.float())
        acc = ((F.sigmoid(x) >= 0.5) == y).float().mean()

        # print(acc)
        # acc = (x.argmax(dim=-1) == y).float().mean()
        return loss, acc

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=0.9)
        # return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def training_step(self, batch, batch_idx):
        # x, y = batch
        tot_loss = 0
        for sz, (x, y) in batch.items():
            loss, acc = self(x, y)
            self.log(f"train_acc_{sz}", acc)
            self.log(f"train_loss_{sz}", loss)
            tot_loss += loss

        tot_loss /= len(batch.items())
        # loss = F.cross_entropy(y_hat, y)
        # acc = (y_hat.argmax(dim=1) == y).float().mean()
        # self.log("train_acc", acc, on_step=False, on_epoch=True)
        # self.log("train_loss", loss, on_step=False, on_epoch=True)
        # y_pred = torch.sigmoid(y_hat)
        # loss = F.binary_cross_entropy(y_pred, y.to(torch.float32))

        # x, y_h, y_v = batch
        # y_logits = self(x)
        # h_prob, v_prob = torch.sigmoid(y_logits[:, :, 0]), torch.sigmoid(y_logits[:, :, 1])
        # loss = F.binary_cross_entropy(h_prob, y_h) + F.binary_cross_entropy(v_prob, y_v)

        # self.train_acc.update(y_pred > 0.5, y)
        # self.log("train_acc", self.train_acc, on_step=False, on_epoch=True)
        # self.log("train_loss", loss, on_step=False, on_epoch=True)

        # x, y_h, _ = batch
        # logit_h, logit_v = self(x), self(x.swapaxes(1, 2))
        # loss = F.binary_cross_entropy_with_logits(logit_h, y_h, dim=1) + F.binary_cross_entropy_with_logits(logit_v, y_v, dim=1)
        # x, y_h, y_v = batch
        # logit_h, logit_v = self(x)
        # return F.binary_cross_entropy_with_logits(logit_h.view(-1, 5*6), y_h)
        # prob_h = torch.sigmoid(logit_h.view(-1, 12))
        # loss = F.binary_cross_entropy(prob_h, y_h)
        # preds = prob_h > 0.5
        return tot_loss

    # def training_epoch_end(self, outputs):
    #     # print(outputs)
    #     # self.log("train_loss", torch.mean(torch.stack([o["loss"] for o in outputs])))
    #     pass

    # def __colate_fn(self, batch):
    #     return F.pad(batch, (0, self.))

    def validation_step(self, batch, batch_idx):
        # print(batch)
        # print(batch.shape)
        tot_loss = 0
        for sz, (x, y) in batch.items():
            loss, acc = self(x, y)
            self.log(f"val_acc_{sz}", acc)
            self.log(f"val_loss_{sz}", loss)
            tot_loss += loss

        tot_loss /= len(batch.items())

        return tot_loss
        # x, y = batch
        # loss, acc = self(x, y)
        # self.log("val_acc", acc)
        # self.log("val_loss", loss)

        # y_hat = self(x).squeeze()
        # loss = F.cross_entropy(y_hat, y)
        # acc = (y_hat.argmax(dim=1) == y).float().mean()
        # self.log("val_acc", acc, on_step=False, on_epoch=True)
        # self.log("val_loss", loss, on_step=False, on_epoch=True)

        # x, y = batch
        # y_hat = self(x).squeeze()
        # y_pred = torch.sigmoid(y_hat)
        # loss = F.binary_cross_entropy(y_pred, y.to(torch.float32))

        # self.val_acc.update(y_pred > 0.5, y)

        # x, y_h, y_v = batch
        # y_logits = self(x)
        # h_prob, v_prob = torch.sigmoid(y_logits[:, :, 0]), torch.sigmoid(y_logits[:, :, 1])
        # loss = F.binary_cross_entropy(h_prob, y_h) + F.binary_cross_entropy(v_prob, y_v)
        # h_preds, v_preds = h_prob > 0.5, v_prob > 0.5
        
        # preds = torch.cat((h_preds.view(-1), v_preds.view(-1)))
        # y = torch.cat((y_h.view(-1), y_v.view(-1)))

        # print(logit_h.shape, y_h.shape)
        # print(y_h)
        # loss = F.cross_entropy(logit_h, y_h)
        # logit_h, logit_v = self(x)#, self(x.swapaxes(-2, -1))
        # loss = F.binary_cross_entropy_with_logits(logit_h.view(-1, 10), y_h)# + F.binary_cross_entropy_with_logits(logit_v, y_v, dim=1)
        # preds = torch.
        # preds = torch.argmax(logit_h, dim=1)
        # self.val_accuracy.update(preds, y)

        # self.log("val_loss", loss, prog_bar=True)
        # self.log("val_acc", self.val_acc, on_step=False, on_epoch=True)
        # self.log("val_loss", loss, on_step=False, on_epoch=True)
        
        # # Update acc with both horizontal and vertical predictions
        # preds = torch.cat([torch.argmax(logit_h, dim=1), torch.argmax(logit_v, dim=1)], dim=0)
        # y = torch.cat([y_h, y_v], dim=0)
        # self.val_accuracy.update(preds, y)

        # # Calling self.log will surface up scalars for you in TensorBoard
        # self.log("val_loss", loss, prog_bar=True)
        # self.log("val_acc", self.val_accuracy, prog_bar=True)

    def test_step(self, batch, batch_idx):
        tot_loss = 0
        for sz, (x, y) in batch.items():
            loss, acc = self(x, y)
            self.log(f"test_acc{sz}", acc)
            tot_loss += loss

        tot_loss /= len(batch.items())
        return tot_loss
            # self.log(f"train_loss_{sz}", loss)
        # x, y = batch
        # # y_hat = self(x).squeeze()
        # _, acc = self(x, y)
        # self.log("test_acc", acc)

        # loss = F.cross_entropy(y_hat, y)
        # acc = (y_hat.argmax(dim=1) == y).float().mean()
        # self.log("train_acc", acc, on_step=False, on_epoch=True)
        # self.log("train_loss", loss, on_step=False, on_epoch=True)

        # x, y = batch
        # y_hat = self(x).squeeze()
        # y_pred = torch.sigmoid(y_hat)
        # loss = F.binary_cross_entropy(y_pred, y.to(torch.float32))

        # return {"loss": loss, "y_pred": y_pred, "y": y}
        # return 
        # self.test_accuracy.update(y_hat > 0.5, y)

        # x, y_h, y_v = batch
        # y_logits = self(x)
        # h_prob, v_prob = torch.sigmoid(y_logits[:, :, 0]), torch.sigmoid(y_logits[:, :, 1])
        # loss = F.binary_cross_entropy(h_prob, y_h) + F.binary_cross_entropy(v_prob, y_v)
        # h_preds, v_preds = h_prob > 0.5, v_prob > 0.5
        
        # preds = torch.cat((h_preds.view(-1), v_preds.view(-1)))
        # y = torch.cat((y_h.view(-1), y_v.view(-1)))
        # x, y_h, _ = batch
        # logit_h = self(x)
        # prob_h = torch.sigmoid(logit_h.view(-1, 12))
        # loss = F.binary_cross_entropy(prob_h, y_h)
        # preds = prob_h > 0.5
        # x, y = batch
        # logits = self(x)
        # loss = F.nll_loss(logits, y)
        # preds = torch.argmax(logits, dim=1)
        # self.test_accuracy.update(preds, y)

        # Calling self.log will surface up scalars for you in TensorBoard
        # self.log("test_loss", loss, prog_bar=True)
        # self.log("test_acc", self.test_accuracy, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    ####################
    # DATA RELATED HOOKS
    ####################

    def prepare_data(self, train_len=8000000, val_len=2000000, test_len=2000000):
        # Generate if not existing.
        for strip in [True, False]:
            for sz in self.train_sz:
                GridIter(self.data_dir, size=sz, strip=strip, generate=True, length=80000, stage='train')
                GridIter(self.data_dir, size=sz, strip=strip, generate=True, length=20000, stage='val')
            for sz in self.test_sz:
                GridIter(self.data_dir, size=sz, strip=strip, generate=True, length=20000, stage='test')

    def setup(self, stage=None):
        # intertwine = lambda a, b: list(itertools.chain.from_iterable(zip(a, b)))
        if stage == "fit" or stage is None:
            # self.ds_train = list(itertools.chain(*[GridIter(self.data_dir, size=sz, stage='train', transform_x=self.transform_x, strip=strip) for sz in self.train_sz for strip in [True, False]]))
            self.ds_train = {sz: list(itertools.chain(*[GridIter(self.data_dir, size=sz, stage='train', transform_x=self.transform_x, strip=strip) for strip in [True, False]])) for sz in self.train_sz}
            
            # self.ds_train = intertwine(iter(GridIter(self.data_dir, size=(5, 6), stage='train', transform_x=self.transform_x, strip=True)),
            #                            iter(GridIter(self.data_dir, size=(5, 6), stage='train', transform_x=self.transform_x, strip=False)))
            # self.ds_val = itertools.chain.from_iterable(GridIter(self.data_dir, size=sz, stage='val', transform_x=self.transform_x, strip=strip) for sz in self.train_sz for strip in [True, False])
            # self.ds_val = list(itertools.chain(*[GridIter(self.data_dir, size=sz, stage='val', transform_x=self.transform_x, strip=strip) for sz in self.train_sz for strip in [True, False]]))
            self.ds_val = {sz: list(itertools.chain(*[GridIter(self.data_dir, size=sz, stage='val', transform_x=self.transform_x, strip=strip) for strip in [True, False]])) for sz in [(5,5), (8, 8)]}
            
            # self.ds_val = intertwine(iter(GridIter(self.data_dir, size=(5, 6), stage='val', transform_x=self.transform_x, strip=True)),
            #                          iter(GridIter(self.data_dir, size=(5, 6), stage='val', transform_x=self.transform_x, strip=False)))
            # self.ds_train = {'strip': {sz: GridIter(self.data_dir, size=sz, stage='train', transform_x=self.transform_x, strip=True) for sz in self.train_sz},
            #                  'no_strip': {sz: GridIter(self.data_dir, size=sz, stage='train', transform_x=self.transform_x, strip=False) for sz in self.train_sz}}
            # self.ds_val = {'strip': {sz: GridIter(self.data_dir, size=sz, stage='val', transform_x=self.transform_x, strip=True) for sz in self.train_sz},
            #                'no_strip': {sz: GridIter(self.data_dir, size=sz, stage='val', transform_x=self.transform_x, strip=False) for sz in self.train_sz}}

        if stage == "test" or stage is None:
            # self.ds_test = itertools.chain.from_iterable(GridIter(self.data_dir, size=sz, stage='test', transform_x=self.transform_x, strip=strip) for sz in self.test_sz for strip in [True, False])
            # self.ds_test = list(itertools.chain(*[GridIter(self.data_dir, size=sz, stage='test', transform_x=self.transform_x, strip=strip) for sz in self.test_sz for strip in [True, False]]))
            self.ds_test = {sz: list(itertools.chain(*[GridIter(self.data_dir, size=sz, stage='test', transform_x=self.transform_x, strip=strip) for strip in [True, False]])) for sz in self.test_sz}
            # self.ds_test = intertwine(iter(GridIter(self.data_dir, size=(5, 6), stage='test', transform_x=self.transform_x, strip=True)),
            #                           iter(GridIter(self.data_dir, size=(5, 6), stage='test', transform_x=self.transform_x, strip=False)))
            # self.ds_test = {'strip': {sz: GridIter(self.data_dir, size=sz, stage='test', transform_x=self.transform_x, strip=True) for sz in self.test_sz},
            #                 'no_strip': {sz: GridIter(self.data_dir, size=sz, stage='test', transform_x=self.transform_x, strip=False) for sz in self.test_sz}}

    def train_dataloader(self):
        # return {strip: {sz: DataLoader(ds, batch_size=self.batch_size) for sz, ds in ds_sz.items()} for strip, ds_sz in self.ds_train.items()}
        # return DataLoader(self.ds_train, batch_size=self.batch_size, pin_memory=True, shuffle=True)
        # return torch_geometric.loader.DataLoader(self.ds_train, batch_size=self.batch_size, pin_memory=True, shuffle=True)
        return CombinedLoader({sz: DataLoader(ds, batch_size=self.batch_size, pin_memory=True, shuffle=True) for sz, ds in self.ds_train.items()})

    def val_dataloader(self):
        # return {strip: {sz: DataLoader(ds, batch_size=self.batch_size) for sz, ds in ds_sz.items()} for strip, ds_sz in self.ds_val.items()}
        # return DataLoader(self.ds_val, batch_size=self.batch_size)
        # return torch_geometric.loader.DataLoader(self.ds_val, batch_size=self.batch_size)
        return CombinedLoader({sz: DataLoader(ds, batch_size=self.batch_size) for sz, ds in self.ds_val.items()})

    def test_dataloader(self):
        # return {strip: {sz: DataLoader(ds, batch_size=self.batch_size) for sz, ds in ds_sz.items()} for strip, ds_sz in self.ds_test.items()}
        # return DataLoader(self.ds_test, batch_size=self.batch_size)
        return CombinedLoader({sz: DataLoader(ds, batch_size=self.batch_size) for sz, ds in self.ds_test.items()})