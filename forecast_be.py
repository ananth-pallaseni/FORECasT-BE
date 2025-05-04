import glob, os
import xgboost as xgb
import numpy as np 
import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqUtils import MeltingTemp as mt

# Model locaton
PACKAGE_DIRECTORY = os.path.dirname(os.path.realpath(__file__))
MODEL_DIR = PACKAGE_DIRECTORY + '/saved_models/'
try:
    assert os.path.exists(MODEL_DIR)
except:
    raise FileNotFoundError(f'Could not find saved models. Looked in: {MODEL_DIR}')

# Models for each position
POS_MODELS = {}

# Models for total editing rate
TOTAL_EFFICIENCY_MODELS = {}

# The mean editing rate parameter is multiplied by these factors to get the editing rate at each position
POS_FRACS = {
    2: 0.05184948248657137,
    3: 0.19673314654134325,
    4: 0.443757534968822,
    5: 0.9119183639518773,
    6: 1.0,
    7: 0.7846071775331374,
    8: 0.41387793695191166,
    9: 0.14388146489658163,
    10: 0.07095196797688848
}

# Map from editor to target base
EDITOR_TARGET_BASE = {
    'CBE': 'C',
    'ABE': 'A'
}

# All valid nucleotides
ALL_NUCS = ['G', 'A', 'C', 'T']

# Wrapper class around xgboost to mimic the releant parts of the sklearn interface without dealing with package version mismatches between xgboost and sklearn
class XGBWrapper:
    def __init__(self):
        self.model = xgb.Booster()
    
    def load(self, path):
        self.model.load_model(path)
    
    def predict(self, x): 
        dmat = xgb.DMatrix(x)
        return self.model.predict(dmat)
        
# Load all positional and total models
def load_models():
    global POS_MODELS
    global TOTAL_EFFICIENCY_MODELS
    editors = [os.path.basename(d) for d in glob.glob(MODEL_DIR + '/*')]
    for editor in editors:
        print(f'Loading models for {editor}')
        model_paths = sorted(glob.glob(MODEL_DIR + f'{editor}/pos*'), key=lambda x: int(x.split('.')[0].split('_')[-1]))
        pos_list = []
        pos_models = []

        # Load model for each position
        for path in model_paths:
            pos = int(path.split('.')[0].split('_')[-1])
            model = XGBWrapper()
            model.load(path)
            pos_list.append(pos)
            pos_models.append(model)
        POS_MODELS[editor] = {}
        POS_MODELS[editor]['pos_list'] = pos_list
        POS_MODELS[editor]['pos_models'] = pos_models

        # Load total efficiency model
        total_model_path = glob.glob(MODEL_DIR + f'{editor}/total*')[0]
        total_model = XGBWrapper()
        total_model.load(total_model_path)
        TOTAL_EFFICIENCY_MODELS[editor] = total_model

    print(f'Finished loading models')

# One hot encode each nucleotide in a sequence
def nucleotide_features(seq):
    feature_labels = []
    features = []
    
    # single nucleotides 
    for pos in range(20):
        nuc = seq[pos]
        for b in ALL_NUCS:
            feature_labels.append(f'{b} at pos {pos+1}')
            features.append(1 if nuc == b else 0)
    
    return features, feature_labels

# Create feature array from a given 20nt sequence
def featurize_20nt_target(target_seq):
    feature_labels = []
    features = []
        
    # target base counts
    for nuc in ALL_NUCS:
        feature_labels.append(f'{nuc} count')
        features.append(sum([n == nuc for n in target_seq]))

    # nucleotide features
    nuc_features, nuc_labels = nucleotide_features(target_seq)
    features += nuc_features
    feature_labels += nuc_labels

    # melting temperatures
    features.append(mt.Tm_NN(target_seq))
    feature_labels.append('Melting Temperature')

    return feature_labels, features

# Scale z-scores for each position to match the given mean & std dev of editing rate
def scale_zscores(zscores, pos_list, mean, std):
    scaled = []
    for z, pos in zip(zscores, pos_list):
        mean_frac = POS_FRACS[pos]
        pos_mean = mean * mean_frac
        if (z is None) or np.isnan(z):
            s = z 
        else:
            s = (z * std) + pos_mean
            s = np.clip(s, 0, 1)
        scaled.append(s)
    return scaled

# For a given sequence and editor type, predict the editing rate at each position 
# Defaults to CBE if no editor provided
# By default, predicts the z-score for the position
# If mean and std provided, then scales the z-score to provide real editing rates between 0-1
def predict(target_seq, mean=None, std=None, editor=None):
    if len(target_seq) != 20:
        raise ValueError(f'Input sequence is {len(target_seq)}, but require 20')
    feature_labels, features = featurize_20nt_target(target_seq)
    features = pd.DataFrame([features], columns=feature_labels) 

    # Get models for editor
    if editor is None:
        editor = 'CBE'
    elif editor not in POS_MODELS:
        raise ValueError(f'Unknown editor: {editor}')
    pos_models = POS_MODELS[editor]['pos_models']
    pos_list = POS_MODELS[editor]['pos_list']

    # Predict per position
    predictions = [model.predict(features)[0] for model in pos_models]
    
    # Rescale if required
    if (mean is not None) and (std is not None):
        predictions = scale_zscores(predictions, pos_list, mean, std)
        
    ret = [(pos, pred) for pos, pred in zip(pos_list, predictions)]
    
    # Overwrite predictions where there is no target base at position
    target_base = EDITOR_TARGET_BASE[editor]
    ret = [(pos, pred) if target_seq[pos-1] == target_base else (pos, None) for pos, pred in zip(pos_list, predictions)]
    return ret

# For a given sequence predict the total rate of editing (fraction of edited reads)
# By default, predicts a z-score
# If mean and std provided, then scales the z-score to provide the real editing rate between 0-1
def predict_total(target_seq, mean=None, std=None, editor=None):
    if len(target_seq) != 20:
        raise ValueError(f'Input sequence is {len(target_seq)}, but require 20')
    feature_labels, features = featurize_20nt_target(target_seq)
    features = pd.DataFrame([features], columns=feature_labels)

    # Get model for editor
    if editor is None:
        editor = 'CBE'
    elif editor not in POS_MODELS:
        raise ValueError(f'Unknown editor: {editor}')
    model = TOTAL_EFFICIENCY_MODELS[editor]

    prediction = model.predict(features)[0]

    # Rescale if required
    if (mean is not None) and (std is not None):
        prediction = scale_zscores([prediction], [6], mean, std)[0]

    return prediction

# Take a fasta file as input and generate positional predictions for each sequence in the file
# Defaults to CBE if no editor provided
# By default, predicts the z-score for the position
# If mean and std provided, then scales the z-score to provide real editing rates between 0-1
def predict_batch_fasta(fasta_path, output_path=None, mean=None, std=None, editor=None):
    # Get models for editor
    if editor is None:
        editor = 'CBE'
    elif editor not in POS_MODELS:
        raise ValueError(f'Unknown editor: {editor}')
    pos_list = POS_MODELS[editor]['pos_list']

    # Predict per position for each guide
    zscore_predictions = []
    guides = []
    for record in SeqIO.parse(fasta_path, 'fasta'):
        if (len(record.seq) != 20) or ('N' in record.seq):
            continue
        preds = [pred for pos, pred in predict(record.seq, editor=editor)]
        zscore_predictions.append(preds)
        guides.append(record.id)

    df = pd.DataFrame(zscore_predictions, index=guides, columns=[f'Pos {pos} Zscore' for pos in pos_list])

    # Scale if required
    if (mean is not None) and (std is not None):
        scaled_predictions = [scale_zscores(p, pos_list, mean, std) for p in zscore_predictions]
        scaled_df = pd.DataFrame(scaled_predictions, index=guides, columns=[f'Pos {pos} Scaled' for pos in pos_list])
        df = pd.concat([df, scaled_df], axis=1)
        df['Mean'] = mean
        df['Std Dev'] = std

    df.index.name = 'Oligo Id'

    if output_path is not None:
        df.to_csv(output_path)

    return df


load_models()
