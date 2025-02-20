import torch
from torch import nn
import torch.nn.functional as F
from .tool import AtomEmbedding, embed_compose
from .encoder import ContextEncoder
from .flow_atom_type import AtomFlow
from .flow_bond_type import BondFlow
from .position_predictor import PositionPredictor
from .pivotal_net import PivotalNet
from easydict import EasyDict
from .EF_calculation import elect_field
from torch_geometric.nn.pool import knn_graph
from torch_geometric.nn import knn, radius
import numpy as np
from torch_scatter import scatter_add,scatter_mean
from .module_blocks import ElectField_transform
from .module_blocks import VNLeakyReLU, VNLinear


encoder_cfg = EasyDict(
    {'edge_channels':8, 'num_interactions':6,
     'knn':32, 'cutoff':10.0}
     )
focal_net_cfg = EasyDict(
    {'hidden_dim_sca':32, 'hidden_dim_vec':8}
    )
atom_flow_cfg = EasyDict(
    {'hidden_dim_sca':32, 'hidden_dim_vec':8, 'num_flow_layers':6}
    )
pos_predictor_cfg = EasyDict(
    {'num_filters':[64,64], 'n_component':3}
    )
pos_filter_cfg = EasyDict(
    {'edge_channels':8, 'num_filters':[32,16]}
    )
edge_flow_cfg = EasyDict(
    {'edge_channels':8, 'num_filters':[32,8], 'num_bond_types':3,
     'num_heads':2, 'cutoff':10.0, 'num_flow_layers':3}
    )
config = EasyDict(
    {'deq_coeff':0.9, 'hidden_channels':32, 'hidden_channels_vec':8, 'bottleneck':8, 'use_conv1d':False,
     'encoder':encoder_cfg, 'atom_flow':atom_flow_cfg, 'pos_predictor':pos_predictor_cfg,
     'pos_filter':pos_filter_cfg, 'edge_flow':edge_flow_cfg, 'focal_net':focal_net_cfg}
    )


class epmolgen(nn.Module):
    def __init__(self, config) -> None:
        super(epmolgen, self).__init__()
        self.config = config
        self.num_bond_types = config.num_bond_types
        self.msg_annealing = config.msg_annealing        
        self.emb_dim = [config.hidden_channels, config.hidden_channels_vec]
        self.protein_atom_emb = AtomEmbedding(config.protein_atom_feature_dim, 1, *self.emb_dim)
        self.ligand_atom_emb = AtomEmbedding(config.ligand_atom_feature_dim, 1, *self.emb_dim)
        self.atom_type_embedding = nn.Embedding(config.num_atom_type, config.hidden_channels)
        self.encoder = ContextEncoder(
            hidden_channels=self.emb_dim, edge_channels=config.encoder.edge_channels, 
            num_edge_types=config.num_bond_types, num_interactions=config.encoder.num_interactions, 
            k=config.encoder.knn, cutoff=config.encoder.cutoff, bottleneck=config.bottleneck,
            use_conv1d=config.use_conv1d, num_heads=config.encoder.num_heads
            )
        self.focal_net = PivotalNet(
            self.emb_dim[0], self.emb_dim[1], config.focal_net.hidden_dim_sca, 
            config.focal_net.hidden_dim_vec, bottleneck=config.bottleneck,
            use_conv1d=config.use_conv1d
            )
        self.atom_flow = AtomFlow(
            self.emb_dim[0], self.emb_dim[1], config.atom_flow.hidden_dim_sca,
            config.atom_flow.hidden_dim_vec, num_lig_atom_type=config.num_atom_type,
            num_flow_layers=config.atom_flow.num_flow_layers, bottleneck=config.bottleneck,
            use_conv1d=config.use_conv1d
            )
        self.pos_predictor = PositionPredictor(
            self.emb_dim[0], self.emb_dim[1], config.pos_predictor.num_filters,
            config.pos_predictor.n_component, bottleneck=config.bottleneck,
            use_conv1d=config.use_conv1d
            )
        
        self.edge_flow = BondFlow(
            self.emb_dim[0], self.emb_dim[1], config.edge_flow.edge_channels, 
            config.edge_flow.num_filters, config.edge_flow.num_bond_types, 
            num_heads=config.edge_flow.num_heads, cutoff=config.edge_flow.cutoff,
            num_st_layers=config.edge_flow.num_flow_layers, bottleneck=config.bottleneck,
            use_conv1d=config.use_conv1d
            )
        self.electric_transform = ElectField_transform(
            num_head=config.encoder.num_heads, 
            hidden_channels=self.emb_dim
            )
        
        self.dimension_transform_sca = nn.Sequential(
            nn.Linear(132, config.atom_flow.hidden_dim_sca), 
            nn.LeakyReLU(), 
            nn.Linear(config.atom_flow.hidden_dim_sca, 64)
        )
        self.dimension_transform_vec = nn.Sequential(
            VNLinear(36, config.atom_flow.hidden_dim_vec, bias=False), 
            VNLeakyReLU(config.atom_flow.hidden_dim_vec), 
            VNLinear(config.atom_flow.hidden_dim_vec, 16, bias=False)
        )
    
    def get_parameter_number(self):                                                                                                                                                         
        total_num = sum(p.numel() for p in self.parameters())
        trainable_num = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {'Total': total_num, 'Trainable': trainable_num}

    def get_loss(self, data):
        h_cpx = embed_compose(data.cpx_feature.float(), data.cpx_pos, data.idx_ligand_ctx_in_cpx, 
                              data.idx_protein_in_cpx, self.ligand_atom_emb, 
                              self.protein_atom_emb, self.emb_dim)
        # encoding context
        ef_edge_index = torch.stack([data.cpx_edge_index[0], data.cpx_edge_index[1]], dim=0)   # code change place
        h_cpx = self.encoder(
            node_attr = h_cpx,
            pos = data.cpx_pos,
            edge_index = data.cpx_edge_index,
            edge_feature = data.cpx_edge_feature,
            annealing=self.msg_annealing
        )
        focal_pred = self.focal_net(h_cpx, data.idx_ligand_ctx_in_cpx)
        focal_loss = F.binary_cross_entropy_with_logits(
            input=focal_pred, target=data.ligand_frontier.view(-1, 1).float()
        )
        focal_pred_apo = self.focal_net(h_cpx, data.apo_protein_idx)
        surf_loss = F.binary_cross_entropy_with_logits(
            input=focal_pred_apo, target=data.candidate_focal_label_in_protein.view(-1, 1).float()
        )
        relative_mu, abs_mu, sigma, pi = self.pos_predictor(
            h_cpx,  # +atom_type_emb[data.step_batch]
            data.focal_idx_in_context,
            data.cpx_pos, 
            atom_type_emb= None
            )
        loss_pos = -torch.log(
            self.pos_predictor.get_mdn_probability(abs_mu, sigma, pi, data.y_pos) + 1e-16
            ).mean()#.clamp_max(10.)  
        
        edge_index_0, edge_index_1 = knn(x=data.cpx_pos, y=data.y_pos, k=32, batch_x = data.step_batch, batch_y = torch.arange(0, data.y_pos.size()[0]).to('cuda:0'))
        ###########################################  Change this depend on which period you are in ###############################
        pretraining = True
        ##########################################################################################################################
        if pretraining == True:
            other_feature_sca = scatter_add(h_cpx[0][edge_index_1], edge_index_0, dim=0)/32
            other_feature_vec = scatter_add(h_cpx[1][edge_index_1], edge_index_0, dim=0)/32
            compose_feature_sca_transform = other_feature_sca   
            compose_feature_vec_transform = other_feature_vec   
        else:
            sca_pos, vec_pos = elect_field(data.cpx_pos[edge_index_1], data.cpx_charge[edge_index_1], data.y_pos[edge_index_0])
            sca_ele = scatter_add(sca_pos, edge_index_0)
            vec_ele = scatter_add(vec_pos, edge_index_0, dim = 0)
            vec_ele = vec_ele.unsqueeze(1)
            sca, vec = self.electric_transform(sca_ele, vec_ele)                                 
            other_feature_sca = scatter_add(h_cpx[0][edge_index_1], edge_index_0, dim=0)/32
            other_feature_vec = scatter_add(h_cpx[1][edge_index_1], edge_index_0, dim=0)/32
            compose_feature_sca = torch.cat((other_feature_sca,sca),dim=1)   
            compose_feature_vec = torch.cat((other_feature_vec,vec),dim=1)   
            sca_focal, vec_focal = h_cpx[0][data.focal_idx_in_context], h_cpx[1][data.focal_idx_in_context]
            sca = torch.cat((sca_focal, compose_feature_sca), dim = 1)
            vec = torch.cat((vec_focal, compose_feature_vec), dim = 1)
            compose_feature_sca_transform = self.dimension_transform_sca(sca)
            compose_feature_vec_transform = self.dimension_transform_vec(vec)

        # for atom loss
        x_z = F.one_hot(data.atom_label, num_classes=self.config.num_atom_type).float() 
        x_z += self.config.deq_coeff * torch.rand(x_z.size(), device=x_z.device) 
        z_atom, atom_log_jacob = self.atom_flow(
            x_z, 
            compose_feature_sca_transform,
            compose_feature_vec_transform)
        ll_atom = (1/2 * (z_atom ** 2) - atom_log_jacob).mean()
        atom_type_emb = self.atom_type_embedding(data.atom_label)      
        
        # for edge loss
        z_edge = F.one_hot(data.edge_label, num_classes=4).float()
        z_edge += self.config.deq_coeff * torch.rand(z_edge.size(), device=z_edge.device)
        edge_index_query = torch.stack([data.edge_query_index_0, data.edge_query_index_1])
        pos_query_knn_edge_idx = torch.stack(
            [edge_index_0, edge_index_1]
            )
        z_edge, edge_log_jacob = self.edge_flow(
            z_edge=z_edge,
            pos_query=data.y_pos, 
            edge_index_query=edge_index_query, 
            cpx_pos=data.cpx_pos, 
            node_attr_compose=h_cpx, 
            edge_index_q_cps_knn=pos_query_knn_edge_idx, 
            index_real_cps_edge_for_atten=data.index_real_cps_edge_for_atten, 
            tri_edge_index=data.tri_edge_index, 
            tri_edge_feat=data.tri_edge_feat,
            atom_type_emb=atom_type_emb,
            annealing=self.msg_annealing
            )
        ll_edge = (1/2 * (z_edge ** 2) - edge_log_jacob).mean()
        loss = torch.nan_to_num(ll_atom)\
             + torch.nan_to_num(loss_pos)\
             + torch.nan_to_num(ll_edge)\
             + torch.nan_to_num(focal_loss)\
             + torch.nan_to_num(surf_loss)\
             
        out_dict = {
            'loss':loss, 'loss_atom':ll_atom, 'loss_edge':ll_edge, #'loss_fake':loss_fake, 'loss_real':loss_real,
            'loss_pos':loss_pos, 'focal_loss':focal_loss, 'surf_loss':torch.nan_to_num(surf_loss)
            }
        return out_dict

