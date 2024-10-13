import os
import numpy as np
from sklearn.utils import resample

FLANKING_LEN = 200

def one_hot_encode(seq):
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}
    seq = seq.upper()
    integer_encoded = np.array([mapping.get(base, 4) for base in seq])
    one_hot_encoded = np.eye(5)[integer_encoded]
    return one_hot_encoded[:, :4]  # Exclude the last column (for 'N')

def get_np_array(oversampling=0, downsampling=0):
    data = {seq_type: [] for seq_type in ['donor', 'acceptor', 'other']}
    
    for seq_type in data:
        for i in range(0, 19305, 1000):
            file_name = f"{seq_type}_seqs_{i}_{i+1000}_flank_{FLANKING_LEN}.txt"
            with open(file_name) as f:
                data[seq_type].extend([line[2:].rstrip() for line in f if line.startswith(">>")])
    
    if oversampling:
        for seq_type in ['donor', 'acceptor']:
            data[seq_type] = resample(data[seq_type], n_samples=oversampling, random_state=1)
    if downsampling:
        data['other'] = resample(data['other'], n_samples=downsampling, random_state=1)
    
    all_seqs = [seq for seqs in data.values() for seq in seqs]
    labels = [i for i, seqs in enumerate(data.values()) for _ in seqs]
    
    X = np.array([one_hot_encode(seq) for seq in all_seqs])
    Y = np.array(labels)
    
    return X, Y

def get_np_array_species(flanking_len, species_name):
    arr_seqs = []
    y_arr = []
    label_map = {'donor': 0, 'acceptor': 1, 'other': 2}
    
    for seq_type in label_map:
        file_name = f"{species_name}_{seq_type}_seqs_flank_{flanking_len}.txt"
        with open(file_name, 'r') as file:
            for line in file:
                if line.startswith(">>"):
                    arr_seqs.append(one_hot_encode(line[2:].strip()))
                    y_arr.append(label_map[seq_type])

    return np.array(arr_seqs), np.array(y_arr)