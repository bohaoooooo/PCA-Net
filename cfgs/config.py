
BASE_PATH = './'
PRETRAIN_MODEL_PATH = os.path.join(BASE_PATH,'pretrain')
DATA_PATH = './ESA_SAR'
TRAIN_DATA_PATH = os.path.join(DATA_PATH)
TRAIN_LABEL_PATH = os.path.join(DATA_PATH)
TRAIN_TXT_PATH = os.path.join(DATA_PATH,'train.txt')
VAL_DATA_PATH = os.path.join(DATA_PATH)
VAL_LABEL_PATH = os.path.join(DATA_PATH)
VAL_TXT_PATH = os.path.join(DATA_PATH,'val.txt')
TEST_DATA_PATH = os.path.join(DATA_PATH)
TEST_TXT_PATH = os.path.join(DATA_PATH,'test.txt')
SAVE_PATH = './output/ESA_SAR'

if not os.path.exists(SAVE_PATH):
    os.mkdir(SAVE_PATH)

INIT_LEARNING_RATE = 0.005
DECAY = 5e-5
MOMENTUM = 0.90
MAX_ITER = 300
BATCH_SIZE = 8
# THRESH = 0.5
THRESHS = [0.1, 0.3, 0.5]
LOSS_PARAM_CONV = 3
LOSS_PARAM_FC = 3
TRANSFROM_SCALES= (160, 160) 
# Choose : LEVIR_CD, WHU_CD, DSIFN_CD
dataset_name = 'ESA_SAR'
# Train with multi-gpu.  Two gpu = [0,1]. Three gpu = [0, 1, 2]
gpu_ids = [0, 1]