import argparse

from argus.model.build import MODEL_REGISTRY

from torch.utils.data import DataLoader, ConcatDataset
from argus.callbacks import MonitorCheckpoint, EarlyStopping
from cnd.ocr.dataset import OcrDataset
from cnd.ocr.argus_model import CRNNModel
from cnd.config import OCR_EXPERIMENTS_DIR, CONFIG_PATH, Config
from cnd.ocr.transforms import get_transforms
from cnd.ocr.metrics import StringAccuracy, StringAccuracyLetters
import string
from pathlib import Path
import torch


torch.backends.cudnn.enabled = False

parser = argparse.ArgumentParser()
parser.add_argument("-en", "--experiment_name", help="Save folder name", required=True)
parser.add_argument("-gpu_i", "--gpu_index", type=str, default="0", help="gpu index")
args = parser.parse_args()

# IF YOU USE GPU UNCOMMENT NEXT LINES:
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_index

# define experiment path
EXPERIMENT_NAME = args.experiment_name
EXPERIMENT_DIR = OCR_EXPERIMENTS_DIR / EXPERIMENT_NAME

CV_CONFIG = Config(CONFIG_PATH)

DATASET_PATHS = [
    Path(CV_CONFIG.get("data_path"))
]
# CHANGE YOUR BATCH SIZE
BATCH_SIZE = 100
# 400 EPOCH SHOULD BE ENOUGH
NUM_EPOCHS = 1500 #400

MODEL_PARAMS = {
    "nn_module":
        ("CRNN", {
            'image_height': CV_CONFIG.get("ocr_image_size")[0],  #As far as h == 1, image height must be equal 16
            'number_input_channels': CV_CONFIG.get("model_image_ch"),  #3 for color image and 1 for gray scale
            'number_class_symbols': len(CV_CONFIG.get('alphabet'))+1,  #Length of alphabet
            'rnn_size': CV_CONFIG.get("model_rnn_size"),  # time length of rnn layer, 64|128|256 and so on
            }),
    "alphabet": CV_CONFIG.get('alphabet'),
    "loss": {"reduction": "mean"},
    "optimizer": ("Adam", {"lr":  0.0001}),  #  0.0001}),
    # CHANGE DEVICE IF YOU USE GPU
    "device": "cpu",
}

if __name__ == "__main__":
    if EXPERIMENT_DIR.exists():
        print(f"Folder 'EXPERIMENT_DIR' already exists")
    else:
        EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)

    transforms = get_transforms(CV_CONFIG.get("ocr_image_size"))  # TODO: define your transforms here
    # define data path

    train_dataset_paths = [p / "train" for p in DATASET_PATHS]

    train_dataset = ConcatDataset([OcrDataset(p, transforms=transforms) for p in train_dataset_paths])  # define your dataset

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=6,
    )
    # IT IS BETTER TO SPLIT DATA INTO TRAIN|VAL AND USE METRICS ON VAL
    val_dataset_paths = [p / "val" for p in DATASET_PATHS]
    val_dataset = ConcatDataset([OcrDataset(p, transforms=transforms) for p in val_dataset_paths])
    #
    val_loader = DataLoader(
         val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4
    )

    model = CRNNModel(MODEL_PARAMS)
    # YOU CAN ADD CALLBACK IF IT NEEDED, FIND MORE IN argus.callbacks
    callbacks = [
        MonitorCheckpoint(EXPERIMENT_DIR, monitor="val_str_accuracy_letter", max_saves=6),
        EarlyStopping(monitor='val_loss', patience=200),
    ]
    # YOU CAN IMPLEMENT DIFFERENT METRICS AND USE THEM TO SEE HOW MANY CORRECT PREDICTION YOU HAVE
    metrics = [StringAccuracy(), StringAccuracyLetters()]

    model.fit(
        train_loader,
        val_loader=val_loader,
        max_epochs=NUM_EPOCHS,
        metrics=metrics,
        callbacks=callbacks,
        metrics_on_train=True,
    )
