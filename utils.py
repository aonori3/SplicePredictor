import os
import numpy as np
import tensorflow as tf
from ensemblrest import EnsemblRest

ensRest = EnsemblRest()

def get_np_array_species(flanking_len, species_name):
    arr_seqs = []
    y_arr = []
    for type_seq in ['donor', 'acceptor', 'other']:
        for i in range(0, 19305, 1000):
            file_name = f"{species_name}_{type_seq}_seqs_flank_{flanking_len}.txt"
            with open(file_name) as fle:
                for line in fle:
                    if line.startswith(">>"):
                        seq = line[2:].rstrip()
                        one_hot_seq = one_hot_encode(seq)
                        arr_seqs.append(one_hot_seq)
                        if type_seq == "donor":
                            y_arr.append(0)
                        elif type_seq == "acceptor":
                            y_arr.append(1)
                        elif type_seq == "other":
                            y_arr.append(2)
    np_arr = np.asarray(arr_seqs)
    np_y_arr = np.asarray(y_arr)
    return np_arr, np_y_arr

def lr_scheduler(epoch, lr):
    decay_rate = 2
    if epoch > 5:
        return lr / decay_rate
    return lr