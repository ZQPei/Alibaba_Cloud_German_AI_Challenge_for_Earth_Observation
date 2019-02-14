DATE = "0212"
DATA_DIR= 'data'
MODEL_DIR = 'model/pkl_all'
OUT_DIR = 'results'
SPECIFIC_NAME = '0214'
USE_CUDA = True
CUDA_VISIBLE_DEVICES = '0'
NETS = ['cbam', 'resnext', 'densenet', 'SENet', 'xception', 'GAC_NET']
NUM_CLASSES = 17
INCHANNELS = 10

SEED = 502

TRAINSET_FILE = 'data/training_0211_07_test4_2.h5'
NETWORK = 'xception'
MIXUP_ALPHA = 1.0
EPOCHES = 200
OPTIMIZER = 'SGD'
LEARNING_RATE = 0.1
MOMENTUM = 0.9
GAMMA = 0.0001
DECAY_METHOD = 'milestone'
DECAY_RATE = 0.1
MILESTONES = [40, 70, 102, 132]
TRAIN_BATCH_SIZE = 32

TESTSET_FILE = 'data/round2_test_b_20190211.h5'
TEST_BATCH_SIZE = 128
USE_TTA = True
TTA_AUG = ['Ori','Ori_Hflip','Ori_Vflip','Ori_Rotate_90','Ori_Rotate_180','Ori_Rotate_270',
            'Crop','Crop_Hflip','Crop_Vflip','Crop_Rotate_90','Crop_Rotate_180','Crop_Rotate_270']
# TTA_AUG = ['Ori'] # uncomment to turn off tta