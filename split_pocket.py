from EPMolGen import Ligand, SplitPocket
from EPMolGen.tools.parse_pdb import Protein
a = Protein("./XXX_protein_prepared.mae", compute_ss= False)
l = Ligand("./XXX_ligand_kekulized.sdf")
result_dict,result_pocket = SplitPocket._split_pocket_with_surface_atoms(a,l,10)
result_dict_string = str(result_dict)
with open("./XXX_topo.dict", "w") as f:
    f.write(result_dict_string)
with open("./XXX_pocket10.pdb", "w") as f:
    f.write(result_pocket)