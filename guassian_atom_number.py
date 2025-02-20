import numpy as np

def sample_atom_num(atom_num_scope, distribution='normal'):
    assert distribution.lower() in {'uniform', 'normal'}, "distribution should be the one of {'uniform', 'normal'}"
    min_atom_num, max_atom_num = atom_num_scope
    if distribution == 'normal':
        mean = int((max_atom_num - min_atom_num) / 2 + min_atom_num)
        std = int((max_atom_num - min_atom_num) / 2)
        num_atom_gen = np.random.normal(mean, std, 1).astype(np.int64).\
            clip(min_atom_num, max_atom_num).item()
        return num_atom_gen
    else:
        return np.random.randint(min_atom_num, max_atom_num, 1)

