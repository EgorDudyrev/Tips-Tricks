import argparse

from argus.model.build import MODEL_REGISTRY
from cnd.ocr.converter import strLabelConverter
from cnd.ocr.predictor import Predictor
from cnd.config import OCR_EXPERIMENTS_DIR, CONFIG_PATH, Config
import imageio
import string
from pathlib import Path
import torch
import os

torch.backends.cudnn.enabled = False

parser = argparse.ArgumentParser()
parser.add_argument("-en", "--experiment_name", help="Save folder name", required=True)
parser.add_argument("-f", "--file_name", help="File name to predict", required=True)
parser.add_argument("-gpu_i", "--gpu_index", type=str, default="0", help="gpu index")
args = parser.parse_args()

# IF YOU USE GPU UNCOMMENT NEXT LINES:
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_index

# define experiment path
EXPERIMENT_NAME = args.experiment_name
EXPERIMENT_DIR = OCR_EXPERIMENTS_DIR / EXPERIMENT_NAME

CV_CONFIG = Config(CONFIG_PATH)


MODEL_PARAMS = {
    "nn_module":
        ("CRNN", {
            'image_height': CV_CONFIG.get('ocr_image_size')[0],  #As far as h == 1, image height must be equal 16
            'number_input_channels': CV_CONFIG.get('model_image_ch'),  #3 for color image and 1 for gray scale
            'number_class_symbols': len(CV_CONFIG.get('alphabet'))+1,  #Length of alphabet
            'rnn_size': CV_CONFIG.get('model_rnn_size'),  # time length of rnn layer, 64|128|256 and so on
            }),
    "alphabet": CV_CONFIG.get('alphabet'),
    "loss": {"reduction": 'mean'},
    "optimizer": ("Adam", {"lr": 0.001}),
    # CHANGE DEVICE IF YOU USE GPU
    "device": "cpu",
}

if __name__ == "__main__":
    converter = strLabelConverter(MODEL_PARAMS['alphabet'])

    model_path = EXPERIMENT_DIR / sorted(os.listdir(EXPERIMENT_DIR))[-1]  # Last saved model
    #print('Model path is', model_path)
    #print('Is it a file?', os.path.isfile(model_path))

    predictor = Predictor(model_path, converter,
                          CV_CONFIG.get('ocr_image_size'), device=MODEL_PARAMS['device'])

    #print('File name', args.file_name)
    #print(os.path.isdir('/workdir/data/CropNumbers'))
    #print(os.path.isfile(args.file_name))
    if os.path.isdir(args.file_name):
        print('DIR Are not supported')
    else:
        img = imageio.imread(args.file_name)
        print(f'Prediction "{predictor.predict(img)}"')
