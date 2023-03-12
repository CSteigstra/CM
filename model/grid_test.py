import argparse
import torch
from pl_base import Model
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import CSVLogger
from tcn.tcn import TCN
from transformer.transformer import Transformer
from vit_pytorch import SimpleViT
import numpy as np
from cnn.cnn import CNN

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

    return x

    # View as if a 2x2 convolution with stride 2.
    x = torch.nn.functional.unfold(x[None, :], (2,2), stride=2).swapaxes(0, 1)

    if not binary:
        # Convert [1, 0, 0, 0] to 0, [0, 1, 0, 0] to 1, etc.
        x = torch.argmax(x, dim=1)

    return x
    # return x.flatten()
    # Maybe flatten
    # return x.view(x.shape[0], -1)

def pad_grid(x):
    x = torch.cat((x[-1:], x), dim=0)
    x = torch.cat((x[:, -1:], x), dim=1)

    # print(x.shape)
    # assert 1 == 2

    return x.view(1, *x.shape)

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

def grid_2_channels(x):
    row, col = x.shape
    x = torch.nn.functional.unfold(x[None, :], (2,2), stride=2).swapaxes(0, 1)
    x = x.argmax(dim=1).view(row//2, col//2, 1).permute(2, 0, 1).float()
    # x = torch.ones((row//2, col//2, 2))
    # x = x.view(row//2, col//2, 4).permute(2, 0, 1)
    # pad_h, pad_v = 4-x.shape[-1], 4-x.shape[-2]
    # r_pad_h, r_pad_v = np.random.randint(0, pad_h+1), np.random.randint(0, pad_v+1)
    # x = torch.nn.functional.pad(x, (pad_h-r_pad_h, r_pad_h, pad_v-r_pad_v, r_pad_v), 'constant', 0)
    # if np.random.rand() < 0.5:
    #     x = x.swapaxes(-1, -2)
    # x = torch.nn.functional.pad(x, (0, 8-x.shape[-1], 0, 8-x.shape[-2]), 'constant', 0)
    return x

# def graph_map(x, wrap=True):
#     h, w = x.shape[0] // 2, x.shape[1] // 2
#     adj = torch.zeros((h*w, h*w))


# def graph_map(x, wrap=True):
#     # Map (i, j) to i * x.shape[1] + j
#     h, w = x.shape[0] // 2, x.shape[1] // 2
#     # adj = torch.zeros((h*w, h*w))

#     # if wrap:
#     #     # Wrap around padding once for convolution.
#     #     x = torch.cat((x[-1:], x, x[:1]), dim=0)
#     #     x = torch.cat((x[:, -1:], x, x[:, :1]), dim=1)

#     # h, w = x.shape
#     # h, w = x.shape[1] // 2, x.shape[2] // 2

#     # View as if a 2x2 convolution with stride 2.
#     # x = torch.nn.functional.unfold(x[None, :], (2,2), stride=2).swapaxes(0, 1)
#     # adj = torch.zeros((x.shape[0], x.shape[0]))

#     adj = torch.zeros((h*w, h*w))

#     for i in range(h):
#         for j in range(w):
#             x_i, x_j = i * 2, j * 2
#             if x[x_i, x_j] == 0:
#                 continue

#             if x[x_i, x_j-1] == 1:
#                 adj[i*w+j, i*w+j-1] = 1
#             if x[x_i-1, x_j] == 1:
#                 adj[i*w+j, (i-1)*w+j] = 1
#             if x[x_i-1, x_j-1] == 1:
#                 adj[i*w+j, (i-1)*w+(j-1)] = 1

#     return adj   

    
# def pad_grid(x, max, rand=False):
#     h_pad, v_pad = 
#     if rand:

#     x = torch.nn.functional.pad(x, (0, max-x.shape[1], 0, max-x.shape[0]), 'constant', 0)
    
#     return x


train_sz, test_sz = [(4, 4)], [(4, 4)] 
# in_size = (train_sz[0] + 1) * (train_sz[1] + 1) * 4


# Transformer model
# base = torch.nn.Transformer(d_model=256, nhead=8, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=2048, dropout=0.1, activation='relu')

# base = Transformer(
# train_sz[0]*train_sz[1], train_sz[0]+train_sz[1], 256, nhead=8, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=2048, dropout=0.1, activation='relu')
# base = Transformer(4, 2, 512, nhead=4, num_encoder_layers=6, dim_feedforward=2048, dropout=0.1, activation='relu')

image_sz = max([max(x) for x in train_sz + test_sz])

# transform_x = lambda x: torch.nn.functional.pad(grid_2_channels(x), (0, image_sz-x.shape[1], 0, image_sz-x.shape[0]), 'constant', 0)

# base = SimpleViT(image_size=image_sz, patch_size=1, num_classes=2, dim=256, depth=6, heads=12, mlp_dim=512, channels=1)
# base = SimpleViT(image_size=image_sz, patch_size=1, num_classes=1, dim=64, depth=4, heads=6, mlp_dim=128, channels=4)

# base = TCN(in_size, train_sz[0]*train_sz[1], [args.nhid] * args.levels, kernel_size=args.ksize, dropout=args.dropout, stride=args.stride)

# model = Model(base, "/mnt/d/AI/thesis/CM/asp_data/small", learning_rate=args.lr, batch_size=args.batch_size, transform_x=grid_2_channels, train_sz=train_sz, test_sz=test_sz)

base = CNN((train_sz[0][0]*2+1, train_sz[0][1]*2+1), 2)

# model = Model(base, "/mnt/d/AI/thesis/CM/asp_data/small", learning_rate=args.lr, batch_size=args.batch_size, transform_x=pad_grid, train_sz=train_sz, test_sz=test_sz)
model = Model(base, ".", learning_rate=args.lr, batch_size=args.batch_size, transform_x=pad_grid, train_sz=train_sz, test_sz=test_sz)


trainer = Trainer(
    accelerator="auto",
    devices=1 if torch.cuda.is_available() else None,  # limiting got iPython runs
    max_epochs=args.epochs,
    callbacks=[TQDMProgressBar(refresh_rate=20)],
    logger=CSVLogger(save_dir="logs/"),
)
trainer.fit(model)

# Test
# model = Model.load_from_checkpoint("logs/lightning_logs/version_46/checkpoints/epoch=24-step=29300.ckpt")
trainer.test(model)
