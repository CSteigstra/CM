import argparse
import torch
from pl_base import Model
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import CSVLogger
from tcn.tcn import TCN
from transformer.transformer import Transformer

parser = argparse.ArgumentParser(description='Sequence Modeling - Grids')
parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                    help='batch size (default: 64)')
parser.add_argument('--cuda', action='store_false',
                    help='use CUDA (default: True)')
parser.add_argument('--dropout', type=float, default=0.05,
                    help='dropout applied to layers (default: 0.05)')
# parser.add_argument('--clip', type=float, default=-1,
#                     help='gradient clip, -1 means no clip (default: -1)')
parser.add_argument('--epochs', type=int, default=20,
                    help='upper epoch limit (default: 20)')
parser.add_argument('--ksize', type=int, default=4,
                    help='kernel size (default: 4)')
parser.add_argument('--stride', type=int, default=4,
                    help='stride (default: 4)')
parser.add_argument('--levels', type=int, default=4,
                    help='# of levels (default: 8)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='report interval (default: 100')
parser.add_argument('--lr', type=float, default=2e-3,
                    help='initial learning rate (default: 2e-3)')
parser.add_argument('--optim', type=str, default='Adam',
                    help='optimizer to use (default: Adam)')
parser.add_argument('--nhid', type=int, default=25,
                    help='number of hidden units per layer (default: 25)')
parser.add_argument('--seed', type=int, default=42,
                    help='random seed (default: 42)')
parser.add_argument('--model', type=str, default='TCN',
                    help='type of model to use (default: TCN)')
args = parser.parse_args()

torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

def conv_wrap(x, wrap=False, binary=True):
    if wrap:
        # Wrap around padding once for convolution.
        x = torch.cat((x[-1:], x, x[:1]), dim=0)
        x = torch.cat((x[:, -1:], x, x[:, :1]), dim=1)

    # View as if a 2x2 convolution with stride 2.
    x = torch.nn.functional.unfold(x[None, :], (2,2), stride=2).swapaxes(0, 1)

    if not binary:
        # Convert [1, 0, 0, 0] to 0, [0, 1, 0, 0] to 1, etc.
        x = torch.argmax(x, dim=1)

    return x.flatten()
    # Maybe flatten
    # return x.view(x.shape[0], -1)

def grid_2_ints(x):
    # Encode each 2x2 subgrid as a single integer.
    # print(x)
    # print(x)
    x = torch.nn.functional.unfold(x[None, :], (2,2), stride=2).swapaxes(0, 1)
    # print(x)
    x = torch.argmax(x, dim=1).flatten()
    # print(x)
    return x

    return torch.argmax(x, dim=1).flatten()

train_sz, test_sz = (1,12), (1, 12) 
# in_size = (train_sz[0] + 1) * (train_sz[1] + 1) * 4


# Transformer model
# base = torch.nn.Transformer(d_model=256, nhead=8, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=2048, dropout=0.1, activation='relu')

# base = Transformer(
# train_sz[0]*train_sz[1], train_sz[0]+train_sz[1], 256, nhead=8, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=2048, dropout=0.1, activation='relu')
base = Transformer(4, 1, 256, nhead=4, num_encoder_layers=4, dim_feedforward=1028, dropout=0.1, activation='relu')

# base = TCN(in_size, train_sz[0]*train_sz[1], [args.nhid] * args.levels, kernel_size=args.ksize, dropout=args.dropout, stride=args.stride)

model = Model(base, ".", learning_rate=args.lr, batch_size=args.batch_size, transform_x=grid_2_ints, train_sz=[train_sz], test_sz=[test_sz])

trainer = Trainer(
    accelerator="auto",
    devices=1 if torch.cuda.is_available() else None,  # limiting got iPython runs
    max_epochs=args.epochs,
    callbacks=[TQDMProgressBar(refresh_rate=20)],
    logger=CSVLogger(save_dir="logs/"),
)
trainer.fit(model)