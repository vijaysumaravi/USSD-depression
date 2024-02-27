import os
import numpy as np

# FEATURE_EXP: logmel, mel, raw, MFCC, MFCC_concat, or text
# WHOLE_TRAIN: This setting is for mitigating the variable length of the data
# by zero padding
# SNV will normalise every file to mean=0 and standard deviation=1
# EXPERIMENT_DETAILS = {'FEATURE_EXP': 'mel',
#                       'FREQ_BINS': 80,
#                       'DATASET_IS_BACKGROUND': False,
#                       'WHOLE_TRAIN': False,
#                       'WINDOW_SIZE': 1024,
#                       'OVERLAP': 50,
#                       'SNV': True,
#                       'SAMPLE_RATE': 16000,
#                       'REMOVE_BACKGROUND': True}

# EXPERIMENT_DETAILS = {'FEATURE_EXP': 'raw',
#                       'FREQ_BINS': 1,
#                       'DATASET_IS_BACKGROUND': False,
#                       'WHOLE_TRAIN': False,
#                       'WINDOW_SIZE': 1024,
#                       'OVERLAP': 50,
#                       'SNV': True,
#                       'SAMPLE_RATE': 16000,
#                       'REMOVE_BACKGROUND': True}

# EXPERIMENT_DETAILS = {'FEATURE_EXP': 'wav2vec2',
#                       'FREQ_BINS': 768,
#                       'DATASET_IS_BACKGROUND': False,
#                       'WHOLE_TRAIN': False,
#                       'WINDOW_SIZE': 1024,
#                       'OVERLAP': 50,
#                       'SNV': True,
#                       'SAMPLE_RATE': 16000,
#                       'REMOVE_BACKGROUND': True}

EXPERIMENT_DETAILS = {'FEATURE_EXP': 'compare16',
                      'FREQ_BINS': 130,
                      'DATASET_IS_BACKGROUND': False,
                      'WHOLE_TRAIN': False,
                      'WINDOW_SIZE': 1024,
                      'OVERLAP': 50,
                      'SNV': True,
                      'SAMPLE_RATE': 16000,
                      'REMOVE_BACKGROUND': True}

# EXPERIMENT_DETAILS = {'FEATURE_EXP': 'rasta_plp',
#                       'FREQ_BINS': 9,
#                       'DATASET_IS_BACKGROUND': False,
#                       'WHOLE_TRAIN': False,
#                       'WINDOW_SIZE': 1024,
#                       'OVERLAP': 50,
#                       'SNV': True,
#                       'SAMPLE_RATE': 16000,
#                       'REMOVE_BACKGROUND': True}


# Set True to split data into genders
GENDER = False
WINDOW_FUNC = np.hanning(EXPERIMENT_DETAILS['WINDOW_SIZE'])
FMIN = 0
FMAX = EXPERIMENT_DETAILS['SAMPLE_RATE'] / 2
HOP_SIZE = EXPERIMENT_DETAILS['WINDOW_SIZE'] -\
           round(EXPERIMENT_DETAILS['WINDOW_SIZE'] * (EXPERIMENT_DETAILS['OVERLAP'] / 100))

if EXPERIMENT_DETAILS['FEATURE_EXP'] == 'text':
    FEATURE_FOLDERS = None
else:
    FEATURE_FOLDERS = ['audio_data', EXPERIMENT_DETAILS['FEATURE_EXP']]

# change these paths to your own
DATASET = '/dataset/daic_woz_old/data' # path to the dataset
WORKSPACE_MAIN_DIR = '/dataset/daic_woz_old/audio_data/feats_DepAudioNet' # path to the folder where data prep will be saved
WORKSPACE_FILES_DIR = '/path/to/USSD-depression/daic_woz_data_prep' # path to where data_processing folder is saved
LABELS_FOLDER='/dataset/daic_woz_old/labels/' # path to the folder where labels are saved
TRAIN_SPLIT_PATH = os.path.join(LABELS_FOLDER, 'train_split_Depression_AVEC2017.csv')
DEV_SPLIT_PATH = os.path.join(LABELS_FOLDER, 'dev_split_Depression_AVEC2017.csv')
TEST_SPLIT_PATH_1 = os.path.join(LABELS_FOLDER, 'test_split_Depression_AVEC2017.csv')
TEST_SPLIT_PATH_2 = os.path.join(LABELS_FOLDER, 'full_test_split.csv')
# CHANGE THIS TO USE A SPECIFIC TEST FILE
TEST_SPLIT_PATH = TEST_SPLIT_PATH_1
FULL_TRAIN_SPLIT_PATH = os.path.join(LABELS_FOLDER, 'full_train_split_Depression_AVEC2017.csv')
COMP_DATASET_PATH = os.path.join(LABELS_FOLDER, 'complete_Depression_AVEC2017.csv')
COMPARE16_FOLDER="/workdir/model_arch/data_processing_depaudionet/comparE16/feats" # path to the folder where compare16 CSV features are saved