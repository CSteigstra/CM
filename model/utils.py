import torch
from torch.utils.data import Dataset, IterableDataset
import pandas as pd
from clyngor import ASP
import os
import numpy as np
from tqdm import tqdm
import itertools

import subprocess
import json
import re

# TODO: Scale N_RUNS with size
N_RUNS = lambda n: int(n / 10000) if n else 0

class zero_y():
    def __init__(self, size):
        self.y = torch.zeros(size)

    def __getitem__(self, _):
        return self.y

# def ASP_runner(lp_file, args, filter='pixel', n=0):
#     args = ['clingo', '--outf=2', lp_file] + args + ['-t', '2', "--rand-freq=1.0"]
#     app = subprocess.Popen(args, stdout=subprocess.PIPE, universal_newlines=True)
#     with tqdm() as pbar:
#         i = 0
#         for line in app.stdout:
#             line = line.lstrip().rstrip().replace('"', '')
#             if line.startswith('pixel'):
#                 atoms = line.split(' ')
#                 x = atoms[:size[0] * size[1]]
#                 y = atoms[size[0] * size[1]:]
#                 yield x, y
#                 i += 1
#                 pbar.update(1)
#                 if i >= n:
#                     break

class ASP_runner():
    def __init__(self, lp_file, args, line_filter='pixel'):
        self.lp_file = lp_file
        self.filter = line_filter
        self.args = args

    def __iter__(self):
        args = ['clingo', '--outf=2', self.lp_file] + self.args + ['-t', '2', "--rand-freq=1.0"]
        app = subprocess.Popen(args, stdout=subprocess.PIPE, universal_newlines=True)
        for line in app.stdout:
            line = line.lstrip().rstrip().replace('"', '')
            if line.startswith(self.filter):
                yield line
        
        app.kill()


class Grid(Dataset):
    def __init__(self, root='.', size=(3, 3), length=0, strip=False, train=True, generate=False):
        self.path = f'{root}/grid_{size[0]}_{size[1]}_{"strip" if strip else "no_strip"}'
        self.root = root
        self.size = size
        self.length = length

        if not os.path.exists(f'{self.path}.csv') and generate:
            self.gen(size, strip)

        self.ds = pd.read_csv(f'{self.path}.csv', delimiter=';', names=['grid', 'strip_h', 'strip_v']).fillna('')

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        row = self.ds.iloc[idx]
        x = self.__transform_x(row['grid'])
        y_h = self.__transform_y(row['strip_h'], self.size[0])
        y_v = self.__transform_y(row['strip_v'], self.size[1])

        return x, y_h, y_v

    def __transform_x(self, x):
        bin_x = torch.zeros(self.size[0] * 2, self.size[1] * 2)
        if x:
            idx = [int(_x) for _x in x.split(',')]
            bin_x[idx[::2], idx[1::2]] = 1

        return bin_x

    def __transform_y(self, y, sz):
        bin_y = torch.zeros(sz)
        if y:
            # TODO: Check mapping of strip_h and strip_v
            idx = [int(_y)-1 for _y in y.split(',')]
            bin_y[idx] = 1

        return bin_y

    def gen(self, size=(3, 3), strip=False, seed=42):
        lp_file = f'{self.root}/strip-mode/gen_{"" if strip else "no_"}strip.lp'
        args = ['-c', f'h={size[0]}', '-c', f'w={size[1]}']

        f_x = open(f'{self.path}.csv', 'w')

        re_pix = re.compile('[0-9]+')
        re_h = re.compile('strip_h\(([0-9]+)\)')
        re_v = re.compile('strip_v\(([0-9]+)\)')

        splits = sorted(np.random.randint(0, self.length + 1, N_RUNS(self.length)))
        n_splits = np.subtract([*splits, self.length], [0, *splits])

        with tqdm() as pbar:
            for n in n_splits:
                cur_args = args + ['-n', str(n), f'--seed={seed}']

                for line in ASP_runner(lp_file, cur_args, 'pixel'):
                    atoms = line.split(' ')
                    x = atoms[:size[0] * size[1]]
                    y = atoms[size[0] * size[1]:]
                    
                    f_x.write(','.join(re_pix.findall(''.join(x))) + ';')
                    if strip:
                        y = atoms[size[0] * size[1]:]
                        f_x.write(','.join(re_h.findall(''.join(y))) + ';')
                        f_x.write(','.join(re_v.findall(''.join(y))) + '\n')
                    else:
                        f_x.write(';\n')

                    pbar.update(1)

                seed += 1

        f_x.close()

class GridIter(IterableDataset):
    def __init__(self, root='.', size=(3, 3), length=0, strip=False, transform_x=None, generate=False, batch_size=1, stage='train'):
        super().__init__()
        if stage in ['train', 'val', 'test']:
            self.path = f'{root}/{stage}/grid_{size[0]}_{size[1]}_{"strip" if strip else "no_strip"}'
        else:
            assert False, f'Unknown stage: {stage}'

        self.root = root
        self.size = size
        self.length = length
        self.batch_size = batch_size

        self.transform_x = transform_x

        if not os.path.exists(f'{self.path}.csv') and generate:
            self.gen(size, strip)

        # self.ds = pd.read_csv(f'{self.path}.csv', delimiter=';', names=['grid', 'strip_h', 'strip_v']).fillna('')

    def __line_mapper(self, line):
        grid, strip_h, strip_v = line.rstrip().split(';')
        return self.__transform_x(grid), self.__transform_hv_bin(strip_h, strip_v)
        return self.__transform_x(grid), *self.__transform_hv(strip_h, strip_v)
        return self.__transform_x(grid), self.__transform_y(strip_h, self.size[1]), self.__transform_y(strip_v, self.size[0])

    def __iter__(self):
        fl = open(f'{self.path}.csv', 'r')
        mapped_itr = map(self.__line_mapper, fl)

        worker = torch.utils.data.get_worker_info()
        if worker is None:
            return mapped_itr

        mapped_itr = itertools.islice(mapped_itr, worker.id, None, self.batch_size)

        # mapped_itr = itertools.islice(mapped_itr, worker.id, None, worker.num_workers)
        
        return mapped_itr

        # worker_total_num = torch.utils.data.get_worker_info().num_workers
        # worker_id = torch.utils.data.get_worker_info().id

        # mapped_itr = itertools.islice(mapped_itr, worker_id, None, worker_total_num)

    # def __getitem__(self, idx):
    #     row = self.ds.iloc[idx]
    #     x = self.__transform_x(row['grid'])
    #     y_h = self.__transform_y(row['strip_h'], self.size[0])
    #     y_v = self.__transform_y(row['strip_v'], self.size[1])

    #     return x, y_h, y_v

    def __transform_x(self, x):
        bin_x = torch.zeros(self.size[0] * 2, self.size[1] * 2)
        if x:
            idx = [int(_x) for _x in x.split(',')]
            bin_x[idx[::2], idx[1::2]] = 1

        if self.transform_x:
            bin_x = self.transform_x(bin_x)
        # print(bin_x.shape)
        # assert 1 == 0

        return bin_x

    def __transform_y(self, y, sz):
        bin_y = torch.zeros(sz)

        if y:
            idx = [int(_y)-1 for _y in y.split(',')]
            bin_y[idx] = 1
    
    def __transform_hv(self, h, v):
        bin_y = torch.zeros((2, *self.size))

        if h:
            idx = [int(_y)-1 for _y in h.split(',')]
            bin_y[0, idx, :] = 1
        if v:
            idx = [int(_y)-1 for _y in v.split(',')]
            bin_y[1, :, idx] = 1

        return bin_y.flatten(1)

    def __transform_hv_bin(self, h, v):
        return 1 if h or v else 0


    def gen(self, size=(3, 3), strip=False, seed=42):
        lp_file = f'{self.root}/strip-mode/gen_{"" if strip else "no_"}strip.lp'
        args = ['-c', f'h={size[0]}', '-c', f'w={size[1]}']

        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        f_x = open(f'{self.path}.csv', 'w')

        re_pix = re.compile('[0-9]+')
        re_h = re.compile('strip_h\(([0-9]+)\)')
        re_v = re.compile('strip_v\(([0-9]+)\)')

        splits = sorted(np.random.randint(0, self.length + 1, N_RUNS(self.length)))
        n_splits = np.subtract([*splits, self.length], [0, *splits])

        with tqdm() as pbar:
            for n in n_splits:
                cur_args = args + ['-n', str(n), f'--seed={np.random.randint(0, 999999999)}']

                for line in ASP_runner(lp_file, cur_args, 'pixel'):
                    atoms = line.split(' ')
                    x = atoms[:size[0] * size[1]]
                    y = atoms[size[0] * size[1]:]
                    
                    f_x.write(','.join(re_pix.findall(''.join(x))) + ';')
                    if strip:
                        y = atoms[size[0] * size[1]:]
                        f_x.write(','.join(re_h.findall(''.join(y))) + ';')
                        f_x.write(','.join(re_v.findall(''.join(y))) + '\n')
                    else:
                        f_x.write(';\n')

                    pbar.update(1)

                seed += 1

        f_x.close()