import copy
from multiprocessing import context
# from multiprocessing import context
import os
import sys
sys.path.append('.')
import random
import time
import uuid
from itertools import compress
# from tqdm import tqdm
import torch
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn.pool import knn_graph
from torch_geometric.transforms import Compose
#from torch_geometric.utils._subgraph import subgraph
from torch_geometric.utils._subgraph import subgraph
from torch_geometric.nn import knn, radius
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_scatter import scatter_add
# import multiprocessing as multi
# from torch_geometric.data import DataLoader


#from EPMolGen.gdbp_model.layers import  Ligand_EF_Calculation
from typing import List, Callable, Union
from torch_geometric.data import Data, HeteroData
from torch_geometric.transforms import BaseTransform
import math


def elect_field(pos, partial_charge, pos_ca):  # pos: 蛋白质原子的位置， partial_charge: 蛋白质原子的电荷, Pos_ca: 小分子的原子的位置
    #bb_pos = torch.as_tensor(bb_pos)#.to('cuda')
    #pos_ca = bb_pos[:,1]

    #pos_all = torch.as_tensor(pos)#.to('cuda')
    #partial_charges = torch.as_tensor(partial_charge)#.to('cuda')
    field_vec = torch.as_tensor(pos_ca).unsqueeze(1) - torch.as_tensor(pos).unsqueeze(0)
    dist = torch.norm(field_vec, dim=-1)
    vec = field_vec / dist.unsqueeze(-1) #torch.nan_to_num()
    ef = vec * partial_charge.view(1, -1, 1) / (20 * math.pi * dist**2).unsqueeze(-1)
    ep = partial_charge / (2 * np.power(10,11) * math.pi * dist)
    v_ef = torch.nan_to_num(ef).sum(-2)
    s_ep = torch.nan_to_num(ep, posinf=0, neginf=0).sum(-1)
    return s_ep, v_ef