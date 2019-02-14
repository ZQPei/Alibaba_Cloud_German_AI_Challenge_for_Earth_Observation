DATE = "0212"
DATA_DIR= 'data'
MODEL_DIR = 'model'
OUT_DIR = 'results'
CUDA_VISIBLE_DEVICES = '0'
NETS = ['cbam', 'resnext', 'densenet', 'SENet', 'xception']
CLASSES = 17
INCHANNELS = 10

TRAIN_FILE = 'data/training_0211_07_test4_2.h5'
MIXUP_ALPHA = 1.0
OPTIMIZER = 'SGD'
LEARNING_RATE = 0.1
MOMENTUM = 0.9
GAMMA = 0.0001
DECAY_METHOD = 'milestone'
DECAY_RATE = 0.1
MILESTONES = [40, 70, 102, 132]
TRAIN_BATCH_SIZE = 32

TEST_FILE = 'data/round2_test_b_20190211.h5'
TEST_BATCH_SIZE = 128
USE_TTA = True