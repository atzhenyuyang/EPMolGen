import torch
#from EPMolGen.gdbp_model import epmolgenWithEdgeNew, reset_parameters, freeze_parameters
from EPMolGen.EPMolGen_model import epmolgen, reset_parameters, freeze_parameters
from EPMolGen.tools.train_finetuning import Experiment
#from EPMolGen.utils.train import Experiment
from EPMolGen.tools import LoadDataset
from EPMolGen.tools.transform import *
#from utils.ParseFile import Protein, parse_sdf_to_dict
from EPMolGen.tools.encode_complex import ComplexData, torchify_dict
import os
from easydict import EasyDict



protein_featurizer = FeaturizeProteinAtom()
ligand_featurizer = FeaturizeLigandAtom(atomic_numbers=[6,7,8,9,15,16,17,35,53])
traj_fn = LigandTrajectory(perm_type='mix', num_atom_type=9)
pivotal_masker = PivotalMasker(r=4, num_work=16, atomic_numbers=[6,7,8,9,15,16,17,35,53])
graph_compose = GraphCompose(
    knn=16, num_workers=16, graph_type='knn', radius=10, use_protein_bond=True
    )
combine = Combine(traj_fn, pivotal_masker, graph_compose)
transform = TrajCompose([
    RefineData(),
    LigandCountNeighbors(),
    protein_featurizer,
    ligand_featurizer,
    combine,
    collate_fn
])


dataset = LoadDataset('/DATA/yzy/EPMolGen_dataset/fine-tuning_dataset.lmdb',transform = transform)
print('Num data:', len(dataset))
train_set, valid_set = LoadDataset.split(dataset, val_num=100, shuffle=True, random_seed=0) 


################################ If you want to do fine-tuning without pretraining #############################################################
"""dataset[0]
## reset parameters
device = 'cuda:0'
#ckpt = torch.load('../path/to/pretrained/ckpt.pt', map_location=device)
ckpt = torch.load('ckpt/ZINC-pretrained-255000.pt', map_location=device)
config = ckpt['config'] 
#model = epmolgenWithEdgeNew(config).to(device)
model = epmolgen(config).to(device) """
############################################################################################################################


device = 'cuda:0'
ckpt = torch.load('./pretraining_ckpt.pt', map_location=device)
config = ckpt['config']
config['encoder']['no_protein'] = False
#config['pos_predictor']['n_component'] = 3

para_name_list = ['pos_predictor.mu_net', 'pos_predictor.logsigma_net', 'pos_predictor.pi_net']

def load_specified_parameters(model, pretrained_ckpt, para_name_list, strict=True):
    pretrained_dict = {}
    for i in pretrained_ckpt['model']:
        for j in para_name_list:
            if j in i:
                break
        else:
            pretrained_dict[i] = pretrained_ckpt['model'][i]
    model_dict = model.state_dict()  
    model_dict.update(pretrained_dict)  
    model.load_state_dict(model_dict, strict=strict) 
    return model

model = epmolgen(config).to(device)
#model = load_specified_parameters(model, ckpt, para_name_list, strict=False)

model.load_state_dict(ckpt['model'], strict = True)
#print(model.get_parameter_number())

keys = ['edge_flow.flow_layers.5', 'atom_flow.flow_layers.5', 
        'pos_predictor.mu_net', 'pos_predictor.logsigma_net', 'pos_predictor.pi_net',
        'focal_net.net.1']
model = reset_parameters(model, keys)


optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=2.e-4, weight_decay=0, betas=(0.99, 0.999))


scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.6, patience=10, min_lr=1.e-5)




exp = Experiment(
    model, train_set, optimizer, valid_set=valid_set, scheduler=scheduler,
    device=device,  use_amp=False
    )


exp.fit_step(
    2000000, valid_per_step=5000, train_batch_size=1, valid_batch_size=1, print_log=True,
    with_tb=True, logdir='./finetuning_log', schedule_key='loss', num_workers=8, 
    pin_memory=False, follow_batch=[], exclude_keys=[], collate_fn=None, 
    max_edge_num_in_batch=2000000
    )