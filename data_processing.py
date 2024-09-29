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
    data = {'donor': [], 'acceptor': [], 'other': []}
    flanking_length = FLANKING_LEN
    
    for type_seq in ['donor', 'acceptor', 'other']:
        for i in range(0, 19305, 1000):
            file_name = f"{type_seq}_seqs_{i}_{i+1000}_flank_{flanking_length}.txt"
            with open(file_name) as fle:
                seqs = [line[2:].rstrip() for line in fle if line.startswith(">>")]
            data[type_seq].extend(seqs)
    
    if oversampling:
        data['donor'] = resample(data['donor'], n_samples=oversampling, random_state=1)
        data['acceptor'] = resample(data['acceptor'], n_samples=oversampling, random_state=1)
    if downsampling:
        data['other'] = resample(data['other'], n_samples=downsampling, random_state=1)
    
    all_seqs = data['donor'] + data['acceptor'] + data['other']
    labels = [0]*len(data['donor']) + [1]*len(data['acceptor']) + [2]*len(data['other'])
    
    X = np.array([one_hot_encode(seq) for seq in all_seqs])
    Y = np.array(labels)
    
    return X, Y

def get_np_array_species(flanking_len, species_name):
    arr_seqs = []
    y_arr = []
    for type_seq in ['donor', 'acceptor', 'other']:
        file_name = f"{species_name}_{type_seq}_seqs_flank_{flanking_len}.txt"
        with open(file_name, 'r') as file:
            for line in file:
                if line.startswith(">>"):
                    seq = line[2:].strip()
                    one_hot_seq = one_hot_encode(seq)
                    arr_seqs.append(one_hot_seq)
                    y_arr.append({'donor': 0, 'acceptor': 1, 'other': 2}[type_seq])

    return np.array(arr_seqs), np.array(y_arr)