import glob, os
from joblib import load
from sklearn.ensemble import GradientBoostingRegressor
from Bio.Seq import Seq
from Bio.SeqUtils import MeltingTemp as mt
import numpy as np 
import pandas as pd

PACKAGE_DIRECTORY = os.path.dirname(os.path.realpath(__file__))
MODEL_DIR = PACKAGE_DIRECTORY + '/saved_models/'
try:
    assert os.path.exists(MODEL_DIR)
except:
    raise FileNotFoundError(f'Could not find saved models. Looked in: {MODEL_DIR}')
POS_MODELS = []
POS_LIST = []

def load_models():
    global POS_MODELS
    model_paths = sorted(glob.glob(MODEL_DIR + '/*'), key=lambda x: int(x.split('.')[0].split('_')[-1]))
    for path in model_paths:
        pos = int(path.split('.')[0].split('_')[-1])
        model = load(path)
        POS_LIST.append(pos)
        POS_MODELS.append(model)

# Microhomology features
def generate_microhomology_matrix(seq):
    base_eq_mat = np.zeros((len(seq), len(seq)))
    for i in list(range(len(seq)))[::-1]:
        for j in list(range(len(seq)))[::-1]:
            eq = seq[i] == seq[j]
            if eq:
                base_eq_mat[i, j] = 1
                if (i < len(seq)-1) and (j < len(seq)-1):
                    base_eq_mat[i, j] += base_eq_mat[i+1, j+1]
    cols = [s for s in seq]
    rows = [s for s in seq]
    bbb = pd.DataFrame(base_eq_mat, columns=cols, index=rows)
    return bbb

def find_microhomology_about_cutsite(seq, pam=20, window=20):
    cutsite = pam-3
    mh_mat = generate_microhomology_matrix(seq).iloc[max(0, cutsite-window):cutsite, cutsite:cutsite+window]
    x, y = np.unravel_index(np.argmax(mh_mat.values, axis=None), mh_mat.shape)
    max_mh_len = mh_mat.iloc[x, y]
    max_dist_between_mh = mh_mat.shape[0]-x + y#cutsite + y - x
    return max_mh_len, max_dist_between_mh

all_nucs = ['A', 'C', 'T', 'G']
all_dinucs = [f'{a}{b}' for a in all_nucs for b in all_nucs]
def nucleotide_features(seq):
    feature_labels = []
    features = []
    
    # single nucleotides 
    for pos in range(20):
        nuc = seq[pos]
        for b in all_nucs:
            feature_labels.append(f'{b} at pos {pos+1}')
            features.append(1 if nuc == b else 0)
            
    # dinucleotides 
    for pos in range(20):
        if pos < 19:
            for dn in all_dinucs:
                feature_labels.append(f'{dn} at pos {pos+1}')
                features.append(1 if seq[pos:pos+2] == dn else 0)
    
    return features, feature_labels

def featurize_20nt_target(target_seq):
    feature_labels = []
    features = []
        
    # target base counts
    for nuc in all_nucs:
        feature_labels.append(f'{nuc} count')
        features.append(sum([n == nuc for n in target_seq]))

    # target GC content
    feature_labels.append('GC content')
    features.append(sum([n in 'GC' for n in target_seq])/len(target_seq))

    # nucleotide features
    nuc_features, nuc_labels = nucleotide_features(target_seq)
    features += nuc_features
    feature_labels += nuc_labels

    # melting temperatures
    features.append(mt.Tm_NN(target_seq))
    feature_labels.append('Melting Temperature')
    for i in range(0, 16):
        features.append(mt.Tm_NN(target_seq[i:i+5]))
        feature_labels.append(f'Melting Temperatute {i+1}-{i+5}')

    # Microhomology
    max_mh_len, max_dist_between_mh = find_microhomology_about_cutsite(target_seq, pam=20, window=20)
    feature_labels += ['Max MH Length', 'Max Distance Between MH']
    features += [max_mh_len, max_dist_between_mh]

    return feature_labels, features

def predict(target_seq, mean=None, std=None):
    feature_labels, features = featurize_20nt_target(target_seq)
    features = np.array(features).reshape(1, -1)

    # Predict per position
    predictions = [model.predict(features)[0] for model in POS_MODELS]
    
    # Rescale if required
    if (mean is not None) and (std is not None):
        predictions = [(pred + mean)*std for pred in predictions]
        predictions = [expit(pred) for pred in predictions]
        
    ret = [(pos, pred) for pos, pred in zip(POS_LIST, predictions)]
    
    # Overwrite predictions where there is no C at position
    ret = [(pos, pred) if target_seq[pos-1] == 'C' else (pos, None) for pos, pred in zip(POS_LIST, predictions)]
    return ret


load_models()