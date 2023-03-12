# import submitit
# # from vit_pytorch import SimpleViT
# from transformer.simple_vit import SimpleViT
from pl_base import Model
from utils import GridIter
import argparse
import torch
from utils import grid_pad, grid_2_channels, grid_2_squares
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from cnn.cnn import CNN
from transformer.simple_vit import SimpleViT
# from transformer.vit import ViT
# from transformer.levit import LeViT
# from transformer.fb_vit import VisionTransformer
# from transformer.crossformer import CrossFormer

# from transformer.mem_transformer import MemTransformerLM

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
parser.add_argument('--epochs', type=int, default=35,
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

# train_sz = [(4,4), (6,6), (7, 7)]
train_sz = [(5,5)]
test_sz = [(8,8)]
# train_sz = [(6, 6), (7, 7)]
# test_sz = [(8,8)]
image_sz = max([max(x) for x in train_sz + test_sz])


# ViT
# base = ViT(image_size=image_sz*2, patch_size=1, num_classes=1, dim=128, depth=3, heads=6, mlp_dim=256, channels=1)
# model = Model(base, "/mnt/d/AI/thesis/CM/asp_data/small", learning_rate=args.lr, batch_size=args.batch_size, transform_x=lambda x: x.view(1, *x.shape), train_sz=train_sz, test_sz=test_sz)

# model


# Transformer-XL (memtransformer)
# base = MemTransformerLM(n_token=4, n_layer=4, n_head=6, d_model=256, d_head=256, d_inner=512)
# model = Model(base, "/mnt/d/AI/thesis/CM/asp_data/small", learning_rate=args.lr, batch_size=args.batch_size, transform_x=grid_pad, train_sz=train_sz, test_sz=test_sz)

# SmallViT
base = SimpleViT(image_size=image_sz*2, patch_size=2, num_classes=1, dim=128, depth=2, heads=6, mlp_dim=128, channels=1, attn_drop_rate=0.05, drop_rate=0.05)
model = Model(base, "data/small", learning_rate=args.lr, batch_size=args.batch_size, transform_x=grid_pad, train_sz=train_sz, test_sz=test_sz)
# model = Model.load_from_checkpoint("tb_logs_simple_vit/lightning_logs/version_9/checkpoints/epoch=5-step=18750.ckpt", model=base)
logger = TensorBoardLogger("tb_logs_simple_vit3/")

# LeViT
# base = LeViT(image_size=image_sz*2, num_classes=1, stages=3, dim=(32, 64, 128), depth=4, heads=(4, 5, 6), mlp_mult=2, dropout=0.1)
# model = Model(base, "/mnt/d/AI/thesis/CM/asp_data/small", learning_rate=args.lr, batch_size=args.batch_size, transform_x=grid_pad, train_sz=train_sz, test_sz=test_sz)

# FB-transfromer with pos-embed interpolation
# base = VisionTransformer(img_size=[image_sz*2], patch_size=2, in_chans=1, num_classes=1, embed_dim=64, depth=5, num_heads=4, qkv_bias=True, drop_rate=0.1, attn_drop_rate=0.1)
# model = Model(base, "/mnt/d/AI/thesis/CM/asp_data/small", learning_rate=args.lr, batch_size=args.batch_size, transform_x=grid_pad, train_sz=train_sz, test_sz=test_sz)
# logger = TensorBoardLogger("tb_logs_fv_vit/")

# CrossFormer
# base = CrossFormer(dim=(32, 64), depth=(2, 4), global_window_size=(2, 2), local_window_size=2, cross_embed_kernel_sizes=((2,4), (2,)), cross_embed_strides=(2, 2), num_classes=1, attn_dropout=0.1, ff_dropout=0.1, channels=1)
# model = Model(base, "/mnt/d/AI/thesis/CM/asp_data/small", learning_rate=args.lr, batch_size=args.batch_size, transform_x=grid_pad, train_sz=train_sz, test_sz=test_sz)
# logger = TensorBoardLogger("tb_logs_cf_vit/")



# VisionTransformer(
#         patch_size=patch_size, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4,
#         qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
#     return model

# CNN
# base = CNN((image_sz * 2 + 1, image_sz * 2 + 1), 1)
# model = Model(base, "/mnt/d/AI/thesis/CM/asp_data/small", learning_rate=args.lr, batch_size=args.batch_size, transform_x=pad_grid, train_sz=train_sz, test_sz=test_sz)

# logger = TensorBoardLogger("tb_logs_levit/")
# logger = CSVLogger(save_dir="logs/")

trainer = Trainer(
    accelerator="auto",
    devices=1 if torch.cuda.is_available() else None,  # limiting got iPython runs
    max_epochs=args.epochs,
    callbacks=[TQDMProgressBar(refresh_rate=20)],
    logger=logger,
)

print(trainer.logger.log_dir)

trainer.fit(model)

# ex = submitit.AutoExecutor("./logs/")

# print("fitting the model?")
# # Test
trainer.test(model)


