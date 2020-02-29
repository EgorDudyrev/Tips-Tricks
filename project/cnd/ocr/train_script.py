import argparse
import os

from torch.utils.data import DataLoader, ConcatDataset

from cnd.ocr.dataset import OcrDataset
from cnd.ocr.model import CRNN
import string
import torch

torch.backends.cudnn.enabled = False

parser = argparse.ArgumentParser()
parser.add_argument("-en", "--experimet_name", help="Save folder name", required=True)
parser.add_argument("-gpu_i", "--gpu_index", type=str, default="0", help="gpu index")
args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_index

#define experiment path
# EXPERIMENT_NAME = args.experimet_name
# EXPERIMENT_DIR = OCR_EXPERIMENTS_DIR / EXPERIMENT_NAME

DATASET_PATHS = [
    "ds1"
]

BATCH_SIZE = 512

# CV_CONFIG =

alphabet = ":,.; "
alphabet += string.ascii_uppercase
alphabet += "".join([str(i) for i in range(10)])

PARAMS = {
    "model": (
        "CRNN",
        {
            "image_height": 10,
            "number_input_channels": 1,
            "number_class_symbols": 32,
            "rnn_size": 128,
        },
    ),
    "loss": {},
    "optimizer": ("Adam", {"lr": 0.001}),
    "alphabet": alphabet,
    "device": "cuda",
}

if __name__ == "__main__":
    # if EXPERIMENT_DIR.exists():
    #     print(f"Folder 'EXPERIMENT_DIR' already exists")
    # else:
    #     EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)

    # transforms =  # define your transforms here
    # define data path
    train_dataset_paths = [p / "train" for p in DATASET_PATHS]
    train_datasets = [
        OcrDataset(p)
        for p in train_dataset_paths
    ]

    train_dataset = ConcatDataset(train_datasets)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=10,
    )

    val_dataset_paths = [p / "val" for p in DATASET_PATHS]
    val_dataset = ConcatDataset([OcrDataset(p) for p in val_dataset_paths])

    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4
    )

    model = CRNN(PARAMS)
    #define callbacks if any
    callbacks = []
    #define metrics if any
    metrics = []

    fit(model,
        train_loader,
        val_loader=val_loader,
        max_epochs=1500,
        metrics=metrics,
        callbacks=callbacks,
        metrics_on_train=True,
    )
