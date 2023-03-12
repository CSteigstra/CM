# import submitit
# # from vit_pytorch import SimpleViT
# from transformer.simple_vit import SimpleViT
from pl_base import Model
from utils import GridIter
import argparse
import torch
from utils import pad_grid, grid_2_channels
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import CSVLogger
from cnn.cnn import CNN
from gnn.gnn import GNNModel
# from grid_test import grid_2_channels, pad_grid



parser = argparse.ArgumentParser(description='Sequence Modeling - Grids')
parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                    help='batch size (default: 64)')
parser.add_argument('--cuda', action='store_false',
                    help='use CUDA (default: True)')
parser.add_argument('--dropout', type=float, default=0.05,
                    help='dropout applied to layers (default: 0.05)')
# parser.add_argument('--clip', type=float, default=-1,
#                     help='gradient clip, -1 means no clip (default: -1)')
parser.add_argument('--epochs', type=int, default=25,
                    help='upper epoch limit (default: 20)')
parser.add_argument('--ksize', type=int, default=4,
                    help='kernel size (default: 4)')
parser.add_argument('--stride', type=int, default=4,
                    help='stride (default: 4)')
parser.add_argument('--levels', type=int, default=4,
                    help='# of levels (default: 8)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='report interval (default: 100')
parser.add_argument('--lr', type=float, default=2e-4,
                    help='initial learning rate (default: 2e-3)')
parser.add_argument('--optim', type=str, default='Adam',
                    help='optimizer to use (default: Adam)')
parser.add_argument('--nhid', type=int, default=25,
                    help='number of hidden units per layer (default: 25)')
parser.add_argument('--seed', type=int, default=42,
                    help='random seed (default: 42)')
parser.add_argument('--model', type=str, default='TCN',
                    help='type of model to use (default: TCN)')
parser.add_argument('--train_sz', type=list, default=[(4,4),(5,5),(6,6)],
                    help='size of training grids (default: [(4,4),(5,5),(6,6)])')
parser.add_argument('--test_sz', type=list, default=[(8,8)],
                    help='size of test grids (default: [(8,8)])')
args = parser.parse_args()

torch.manual_seed(args.seed)

train_sz = [(8,8)]
test_sz = [(8,8)]
image_sz = max([max(x) for x in train_sz + test_sz])


# GNN
base = GNNModel(4, 64, 1, 3)
model = Model(base, "/mnt/d/AI/thesis/CM/asp_data/small", learning_rate=args.lr, batch_size=args.batch_size, transform_x=grid_2_channels, train_sz=train_sz, test_sz=test_sz)


# ViT
# base = ViT(image_size=image_sz*2, patch_size=1, num_classes=1, dim=128, depth=3, heads=6, mlp_dim=256, channels=1)
# model = Model(base, "/mnt/d/AI/thesis/CM/asp_data/small", learning_rate=args.lr, batch_size=args.batch_size, transform_x=lambda x: x.view(1, *x.shape), train_sz=train_sz, test_sz=test_sz)

# model


# SmallViT
# base = SimpleViT(image_size=image_sz*2, patch_size=1, num_classes=1, dim=128, depth=3, heads=6, mlp_dim=256, channels=1)
# model = Model(base, "/mnt/d/AI/thesis/CM/asp_data/small", learning_rate=args.lr, batch_size=args.batch_size, transform_x=lambda x: x.view(1, *x.shape), train_sz=train_sz, test_sz=test_sz)

# CNN
# base = CNN((image_sz * 2 + 1, image_sz * 2 + 1), 1)
# model = Model(base, "/mnt/d/AI/thesis/CM/asp_data/small", learning_rate=args.lr, batch_size=args.batch_size, transform_x=pad_grid, train_sz=train_sz, test_sz=test_sz)

trainer = Trainer(
    accelerator="auto",
    devices=1 if torch.cuda.is_available() else None,  # limiting got iPython runs
    max_epochs=args.epochs,
    callbacks=[TQDMProgressBar(refresh_rate=20)],
    logger=CSVLogger(save_dir="logs/"),
)

print(trainer.logger.log_dir)

# ex = submitit.AutoExecutor("./logs/")

trainer.fit(model)
print("fitting the model?")
# # Test
trainer.test(model)


