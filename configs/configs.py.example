import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CLIP_VIT_B16_PATH = os.getenv('CLIP_VIT_B16_PATH', str(PROJECT_ROOT / 'models' / 'ViT-B-16.pt'))
CLIP_VIT_B32_PATH = os.getenv('CLIP_VIT_B32_PATH', str(PROJECT_ROOT / 'models' / 'ViT-B-32.pt'))
CLIP_VIT_L14_PATH = os.getenv('CLIP_VIT_L14_PATH', str(PROJECT_ROOT / 'models' / 'ViT-L-14.pt'))


DWCONV3D_DISABLE_CUDNN = True


DATASETS = {
    'samdd': dict(
        TRAIN_ROOT=os.getenv('SAMDD_ROOT', 'path/to/SAM-DD'),
        VAL_ROOT=os.getenv('SAMDD_ROOT', 'path/to/SAM-DD'),
        TRAIN_LIST=str(PROJECT_ROOT / 'lists' / 'samdd'),
        VAL_LIST=str(PROJECT_ROOT / 'lists' / 'samdd'),
        NUM_CLASSES=10,
        N_FOLDS=6,
    ),
    'samdd_single': dict(
        TRAIN_ROOT=os.getenv('SAMDD_ROOT', 'path/to/SAM-DD'),
        VAL_ROOT=os.getenv('SAMDD_ROOT', 'path/to/SAM-DD'),
        TRAIN_LIST=str(PROJECT_ROOT / 'lists' / 'samdd_single'),
        VAL_LIST=str(PROJECT_ROOT / 'lists' / 'samdd_single'),
        NUM_CLASSES=8,
        N_FOLDS=6,
    ),
    'driveract_16': dict(
        TRAIN_ROOT=os.getenv('DRIVERACT16_ROOT', 'path/to/DriverAct_16'),
        VAL_ROOT=os.getenv('DRIVERACT16_ROOT', 'path/to/DriverAct_16'),
        TRAIN_LIST=str(PROJECT_ROOT / 'lists' / 'driveract_16'),
        VAL_LIST=str(PROJECT_ROOT / 'lists' / 'driveract_16'),
        NUM_CLASSES=9,
        N_FOLDS=5,
    ),
    'dmd': dict(
        TRAIN_ROOT=os.getenv('DMD_ROOT', 'path/to/DMD'),
        VAL_ROOT=os.getenv('DMD_ROOT', 'path/to/DMD'),
        TRAIN_LIST=str(PROJECT_ROOT / 'lists' / 'dmd'),
        VAL_LIST=str(PROJECT_ROOT / 'lists' / 'dmd'),
        NUM_CLASSES=10,
        N_FOLDS=5,
    ),
    'dmd_lowlight': dict(
        TRAIN_ROOT=os.getenv('DMD_ROOT', 'path/to/DMD'),
        VAL_ROOT=os.getenv('DMD_LOWLIGHT_ROOT', 'path/to/DMD_lowlight'),
        TRAIN_LIST=str(PROJECT_ROOT / 'lists' / 'dmd'),
        VAL_LIST=str(PROJECT_ROOT / 'lists' / 'dmd_lowlight'),
        NUM_CLASSES=10,
        N_FOLDS=5,
    ),
}
