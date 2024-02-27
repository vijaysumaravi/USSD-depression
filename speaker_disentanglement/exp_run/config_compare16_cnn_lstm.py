import os

# Use this string to write a brief detail about the current experiment. This
# string will be saved in a logger for this particular experiment
EXPERIMENT_BRIEF = ''

# Set to complete to use all the data
# Set to sub to use training/dev sets only
# Network options: custom or custom_att (to use the attention mechanism)
# Set to complete to use all the data
# Set to sub to use training/dev sets only
# Network options: custom or custom_att (to use the attention mechanism)
EXPERIMENT_DETAILS = {'FEATURE_EXP': 'compare16_delta',
                      'CLASS_WEIGHTS': False,
                      'USE_GENDER_WEIGHTS': False,
                      'SUB_SAMPLE_ND_CLASS': True,  # Make len(dep) == len(
                      # ndep)
                      'CROP': True,
                      'OVERSAMPLE': False,
                      'SPLIT_BY_GENDER': False,  # Only for use in test mode
                      'FEATURE_DIMENSIONS': 384, #120 for mel, 1 for contentvec/wavlm(+)/whisper, 61440 for raw, 200 for wav2vec2, 384 for compare16
                      'PRETRAIN_STEPS': 193, #not used if mel / raw / wav2vec2
                      'PRETRAIN_DIMS': 768, #not used if mel/ raw / wav2vec2, 768 for contentvec/wavlm, 512 for whisper_base
                      'pretrained_layer': 13, #not used if not multi layer rep
                      'FREQ_BINS': 130, #40 / 80 for mel, Not used for contentvec/wavlm, 1 for raw, 768 for wav2vec2, 130 for compare16
                      'BATCH_SIZE': 20,
                      'SNV': True,
                      'LEARNING_RATE': 3e-3,
                      'weight_decay': 0,
                      'lrf': 2,
                      'LOSS_ALPHA': 5e-5, #loss weight for speaker 
                      'SEED': 1000,
                      'TOTAL_EPOCHS': 100,
                      'TOTAL_ITERATIONS': 3280,
                      'ITERATION_EPOCH': 1,
                      'EXP_RUNTHROUGH': 5}
# Determine the level of crop, min file found in training set or maximum file
# per set (ND / D) or (FND, MND, FD, MD)
MIN_CROP = True
# Determine whether the experiment is run in terms of 'epoch' or 'iteration'
ANALYSIS_MODE = 'epoch'

#EXPERIMENT_DETAILS['SUB_DIR']= 'contentvec_debug/'

#model: ecapa_tdnn_small (for mel / raw) / wav2vec (for contentvec / wavlm)
model_type = 'cnn_lstm' #'ecapa_tdnn_small' or 'wav2vec' or 'cnn_lstm'

#disentanglement_method: speaker loss type 'adv', 'LE', 'LE_var', 'LE_KL', 'LE_KL_var', 'LE_KL_1', 'cos_similarity', 'cos_similarity_min'
dist_type = 'cos_similarity_min'

# spk_embd_dimension 128, 192, 256 -> 192 for hf,  128 for hf + pca,  256 for xvec + pca 
spk_embd_dim = 256 

# if EXPERIMENT_DETAILS['LOSS_ALPHA'] == 0:
#     EXPERIMENT_DETAILS['SUB_DIR'] = ('vijay/utterance/'+str(spk_embd_dim)+'_dim/'+EXPERIMENT_DETAILS['FEATURE_EXP']+'/'+model_type
#                                 + "_feature_"+EXPERIMENT_DETAILS["FEATURE_EXP"]
#                                 +"_feat_dim_"+str(EXPERIMENT_DETAILS["FEATURE_DIMENSIONS"])
#                                 + '_batch_'+str(EXPERIMENT_DETAILS['BATCH_SIZE'])
#                                 + '_lr_'+str(EXPERIMENT_DETAILS['LEARNING_RATE'])
#                                 + '_wdecay_'+str(EXPERIMENT_DETAILS['weight_decay']) 
#                                 + '_lrf_'+str(EXPERIMENT_DETAILS['lrf'])
#                                 + '/')
# else:
#      EXPERIMENT_DETAILS['SUB_DIR'] = ('vijay/utterance/'+str(spk_embd_dim)+'_dim/'+EXPERIMENT_DETAILS['FEATURE_EXP']+'_dist_'+dist_type+'/'+model_type
#                                 + "_feature_"+EXPERIMENT_DETAILS["FEATURE_EXP"]
#                                 +"_feat_dim_"+str(EXPERIMENT_DETAILS["FEATURE_DIMENSIONS"])
#                                 + '_batch_'+str(EXPERIMENT_DETAILS['BATCH_SIZE'])
#                                 + '_lr_'+str(EXPERIMENT_DETAILS['LEARNING_RATE'])
#                                 + '_wdecay_'+str(EXPERIMENT_DETAILS['weight_decay']) 
#                                 + '_lrf_'+str(EXPERIMENT_DETAILS['lrf'])
#                                 + '_alpha_'+str(EXPERIMENT_DETAILS['LOSS_ALPHA'])
#                                 + '/')


EXPERIMENT_DETAILS['SUB_DIR'] = 'cnn_lstm_feature_compare16_delta_feat_dim_384_batch_20_lr_0.003_wdecay_0_lrf_2_alpha_4e-06' # save this folder in /Dataset/daic-woz-old/audio_feats/feats_DepAudioNet/compare16_delta_snv_exp

# How to calculate the weights: 'macro' uses the number of individual
# interviews in the training set (e.g. 31 dep / 76 non-dep), 'micro' uses the
# minimum number of segments of both classes (e.g. min_num_seg_dep=35,
# therefore every interview in depressed class will be normalised according
# to 35), 'both' combines the macro and micro via the product, 'instance'
# uses the total number of segments for each class to determine the weights (
# e.g. there could be 558 dep segs and 440 non-dep segs).
WEIGHT_TYPE = 'instance'

# Set to 'm' or 'f' to split into male or female respectively
# Otherwise set to '-' to keep both genders in the database
GENDER = '-'

# These values should be the same as those used to create the database
# If raw audio is used, you might want to set these to the conv kernel and
# stride values
WINDOW_SIZE = 1024
HOP_SIZE = 512
OVERLAP = int((HOP_SIZE / WINDOW_SIZE) * 100)

FEATURE_FOLDERS = ['audio_data', 'logmel']
EXP_FOLDERS = ['log', 'model', 'condor_logs']

if EXPERIMENT_DETAILS['FEATURE_EXP'] == 'text':
    FEATURE_FOLDERS = None
else:
    FEATURE_FOLDERS = ['audio_data', 'logmel']
EXP_FOLDERS = ['log', 'model', 'condor_logs']

if EXPERIMENT_DETAILS['FEATURE_EXP'] == 'logmel' or EXPERIMENT_DETAILS[
    'FEATURE_EXP'] == 'MFCC' or EXPERIMENT_DETAILS['FEATURE_EXP'] == \
        'MFCC_concat':
    if EXPERIMENT_DETAILS['DATASET_IS_BACKGROUND']:
        FOLDER_NAME = f"BKGND_{EXPERIMENT_DETAILS['FEATURE_EXP']}" \
                      f"_{str(EXPERIMENT_DETAILS['FREQ_BINS'])}"
    elif not EXPERIMENT_DETAILS['DATASET_IS_BACKGROUND'] and \
            EXPERIMENT_DETAILS['REMOVE_BACKGROUND']:
        FOLDER_NAME = f"{EXPERIMENT_DETAILS['FEATURE_EXP']}_{str(EXPERIMENT_DETAILS['FREQ_BINS'])}"
    elif not EXPERIMENT_DETAILS['DATASET_IS_BACKGROUND'] and not \
            EXPERIMENT_DETAILS['REMOVE_BACKGROUND']:
        FOLDER_NAME = f"{EXPERIMENT_DETAILS['FEATURE_EXP']}" \
                      f"_" \
                      f"{str(EXPERIMENT_DETAILS['FREQ_BINS'])}_with_backgnd"
else:
    FOLDER_NAME = f"{EXPERIMENT_DETAILS['FEATURE_EXP']}"

if EXPERIMENT_DETAILS['SNV']:
    FOLDER_NAME = FOLDER_NAME + '_snv_exp'
else:
    FOLDER_NAME = FOLDER_NAME + '_exp'

if EXPERIMENT_DETAILS['USE_GENDER_WEIGHTS']:
    EXPERIMENT_DETAILS['SUB_DIR'] = EXPERIMENT_DETAILS['SUB_DIR'] + '_gen'


DATASET = '/path/to/daic_woz_dataset/daic-woz-old/data'
WORKSPACE_MAIN_DIR = '/path/to/daic_woz_dataset/daic-woz-old/audio_feats/feats_DepAudioNet/' 
WORKSPACE_FILES_DIR = '/path/to/USSD-depression/speaker_disentanglement'
TRAIN_SPLIT_PATH = os.path.join(DATASET, 'train_split_Depression_AVEC2017.csv')
DEV_SPLIT_PATH = os.path.join(DATASET, 'dev_split_Depression_AVEC2017.csv')
# TEST_SPLIT_PATH = os.path.join(DATASET, 'test_split_Depression_AVEC2017.csv')
TEST_SPLIT_PATH = os.path.join(DATASET, 'full_test_split.csv')
FULL_TRAIN_SPLIT_PATH = os.path.join(DATASET, 'full_train_split_Depression_AVEC2017.csv')
COMP_DATASET_PATH = os.path.join(DATASET, 'complete_Depression_AVEC2017.csv')
spk_embd_path = '/path/to/spk_embeddings_kaldi_256.pkl'

