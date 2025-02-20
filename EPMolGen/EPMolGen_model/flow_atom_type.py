import torch
from torch import nn
from torch.nn import functional as F
from .module_blocks import GDBPerceptronVN, GDBLinear, ST_GDBP_Exp
from .module_blocks import VNLinear, VNLeakyReLU

class AtomFlow(nn.Module):
    def __init__(self, in_sca, in_vec, hidden_dim_sca, hidden_dim_vec, num_lig_atom_type=10,
                 num_flow_layers=6, bottleneck=1, use_conv1d=False) -> None:
        super(AtomFlow, self).__init__()
        
        self.net = nn.Sequential(
            GDBPerceptronVN(
                in_sca, in_vec, hidden_dim_sca, hidden_dim_vec, bottleneck=bottleneck, use_conv1d=use_conv1d
                ),
            GDBLinear(
                hidden_dim_sca, hidden_dim_vec, hidden_dim_sca, hidden_dim_vec, bottleneck=bottleneck, 
                use_conv1d=use_conv1d
                )
        )
        
        self.flow_layers = nn.ModuleList()
        for _ in enumerate(range(num_flow_layers)):
            layer = ST_GDBP_Exp(
                hidden_dim_sca, hidden_dim_vec, num_lig_atom_type, hidden_dim_vec, bottleneck=bottleneck,
                use_conv1d=use_conv1d
            )
            self.flow_layers.append(layer)

    def forward(self, z_atom, sca_feature, vec_feature):
        sca_atom, vec_atom = self.net([sca_feature, vec_feature])
        for ix in range(len(self.flow_layers)):
            s, t = self.flow_layers[ix]([sca_atom, vec_atom])
            s = s.exp()
            z_atom = (z_atom + t) * s
            if ix == 0:
                atom_log_jacob = (torch.abs(s) + 1e-20).log()
            else:
                atom_log_jacob += (torch.abs(s) + 1e-20).log()
        return z_atom, atom_log_jacob

    def reverse(self, atom_latent, sca_feature, vec_feature):
        sca_atom, vec_atom = self.net([sca_feature, vec_feature])
        for ix in range(len(self.flow_layers)):
            s, t = self.flow_layers[ix]([sca_atom, vec_atom])
            atom_latent = (atom_latent / s.exp()) - t
        return atom_latent