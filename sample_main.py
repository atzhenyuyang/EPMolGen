import argparse
from EPMolGen import epmolgen, Generate
from EPMolGen.tools import *
from EPMolGen.tools import mask_node, ComplexData
from EPMolGen.tools.parse_pdb import Protein

from guassian_atom_number import sample_atom_num

def str2bool(v):
    if v.lower() in {'yes', 'true', 't', 'y', '1'}:
        return True
    elif v.lower() in {'no', 'False', 'f', 'n', '0'}:
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parameter():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, default=' ', help='the path of saved model')
    parser.add_argument('--max_atom_num', type=int, default=40, help='the max atom number for generation')
    parser.add_argument('-pit', '--pivotal_threshold', type=float, default=0.5, help='the threshold of probility for pivotal atom')
    parser.add_argument('-cm', '--choose_max', type=str, default='1', help='whether choose the atom that has the highest prob as pivotal atom')
    parser.add_argument('--min_dist_inter_mol', type=float, default=3.0, help='inter-molecular dist cutoff between protein and ligand.')
    parser.add_argument('--bond_length_range', type=str, default=(1.0,2.0), help='the range of bond length for mol generation.')
    parser.add_argument('-pkt', '--pocket', type=str, default='None', help='the pdb file of pocket in receptor')
    parser.add_argument('-topo', type=str, default='None', help='the topology file of the protein pocket')
    parser.add_argument('-n', '--num_gen', type=int, default=100, help='the number of molecules to be generated')
    parser.add_argument('--name', type=str, default='receptor', help='receptor name')
    parser.add_argument('-d', '--device', type=str, default='cuda:0', help='cuda:x or cpu')
    parser.add_argument('-at', '--atom_temperature', type=float, default=1.0, help='temperature for atom sampling')
    parser.add_argument('-bt', '--bond_temperature', type=float, default=1.0, help='temperature for bond sampling')
    parser.add_argument('-mdb', '--max_double_in_6ring', type=int, default=0, help='')
    parser.add_argument('--with_print', type=str, default='1', help='whether print SMILES in generative process')
    parser.add_argument('--root_path', type=str, default='gen_results', help='the root path for saving results')
    parser.add_argument('--readme', '-rm', type=str, default='None', help='description of this genrative task')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    args = parameter()
    if args.name == 'receptor':
        args.name = args.pocket.split('/')[-1].split('-')[0]
###################    For your convenience, you can test your code by implemented it here    ##############################
    args.pocket = './YTHDC2/6k6u_pocket10.pdb' 
    args.ckpt = './model_ckpt.pt'  
    args.choose_max = 0
    args.num_gen = 5000
    args.with_print = True
    args.choose_max = 1
    args.atom_temperature = 1.0
    args.bond_temperature = 1.0
    args.name = 'YTHDC2' 
    args.root_path = 'gen_results' 
    args.device = 'cuda:0'
    args.topo = './YTHDC2/6k6u_topo.dict'
###################      For your convenience, you can test your code by implemented it here      ##############################
    ## Load Target
    assert args.pocket != 'None', 'Please specify pocket !'
    assert args.ckpt != 'None', 'Please specify model !'
    device = args.device
    pdb_file = args.pocket
    pro_dict = Protein(pdb_file, topo_info = args.topo).get_atom_dict(removeHs=False)
    lig_dict = Ligand.empty_dict()
    atomic_numbers=[6,7,8,9,15,16,17,35,53]
    encode_complex = ComplexData.from_protein_ligand_dicts(
                    protein_dict=torchify_dict(pro_dict),
                    ligand_dict=torchify_dict(lig_dict),
                )
    protein_featurizer = FeaturizeProteinAtom()
    ligand_featurizer = FeaturizeLigandAtom(atomic_numbers=atomic_numbers)
    pivotal_masker = PivotalMasker(r=6.0, num_work=16, atomic_numbers=atomic_numbers)
    graph_compose = GraphCompose(knn=16, num_workers=16, for_gen=True, use_protein_bond=True)
    encode_complex = RefineData()(encode_complex)
    encode_complex = LigandCountNeighbors()(encode_complex)
    encode_complex = protein_featurizer(encode_complex)
    encode_complex = ligand_featurizer(encode_complex)
    node4mask = torch.arange(encode_complex.ligand_pos.size(0))
    encode_complex = mask_node(encode_complex, torch.empty([0], dtype=torch.long), node4mask, num_atom_type=9, y_pos_std=0.)
    encode_complex = graph_compose.run(encode_complex)

    ## Load model
    print('Loading model ...')
    ckpt = torch.load(args.ckpt, map_location=device)
    config = ckpt['config']
    model = epmolgen(config).to(device)
    model.load_state_dict(ckpt['model'])
    print('Generating molecules ...')
    temperature = [args.atom_temperature, args.bond_temperature]
    if isinstance(args.bond_length_range, str):
        args.bond_length_range = eval(args.bond_length_range)
    sample= Generate(model, graph_compose.run, temperature=temperature, atom_type_map=[6,7,8,9,15,16,17,35,53],
                        num_bond_type=4, max_atom_num=35 , pivotal_threshold=args.pivotal_threshold, 
                        max_double_in_6ring=args.max_double_in_6ring, min_dist_inter_mol=args.min_dist_inter_mol,
                        bond_length_range=args.bond_length_range, choose_max=args.choose_max, device=device)
    sample.generate(encode_complex, num_gen=args.num_gen, rec_name=args.name, with_print=args.with_print,
                      root_path=args.root_path)
    os.system('cp {} {}'.format(args.ckpt, sample.out_dir))