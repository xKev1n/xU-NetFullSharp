import argparse
from utils import eval_results, test_model, load_model, load_weights, get_flops, compile_model, eval_test_results, eval_test_results_woPred
from models.UNet3P import UNet3P
from models.UNetPP import UNetPP
from models.UNet import UNet
from models.Att_UNet import Att_UNet
from models.Att_UNetPP import Att_UNetPP
from models.DeepResUNet import DeepResUNet
from models.UNetSharp import UNetSharp
from models.DeBoNet import DeBoNet
from models.Kalisz_AE import KaliszAE
from models.xUNetFS import xUNetFS
from models.Att_xUNetFS import Att_xUNetFS

import tensorflow as tf
import numpy as np
import os
import json

# Load the JSON file
with open("weights_mapping.json", "r") as file:
    weights_mapping = json.load(file)

# Function to get weights path by model name
def get_weights_path(model_name):
    try:
        return weights_mapping[model_name]
    except KeyError:
        raise ValueError(f"Model {model_name} is not recognized.\nAvailable models: {list(weights_mapping.keys())}")

def main(args):
    # Model selection based on input argument
    
    model_name = args.model_name
    
    if model_name == "UNET_RES18":
        model = DeBoNet(COMPILE=False, NAME=model_name)
    if model_name == "FPN_RES18":
        model = DeBoNet(COMPILE=False, NAME=model_name)
    if model_name == "FPN_EF0":
        model = DeBoNet(COMPILE=False, NAME=model_name)
    elif model_name == "KALISZ_AE":
        model = KaliszAE()
    elif model_name == "UNET3P":
        model = UNet3P()
    elif model_name == "UNETPP":
        model = UNetPP()
    elif model_name == "UNET":
        model = UNet()
    elif model_name == "ATT_UNET":
        model = Att_UNet()
    elif model_name == "ATT_UNETPP":
        model = Att_UNetPP()
    elif model_name == "DEEP_RESUNET":
        model = DeepResUNet()
    elif model_name == "UNET_SHARP":
        model = UNetSharp()
    elif model_name == "XUNETFS":
        model = xUNetFS()
    elif model_name == "ATT_XUNETFS":
        model = Att_xUNetFS()
    else:
        raise ValueError(f"Model {model_name} is not recognized.\nAvailable models: {list(weights_mapping.keys())}")

    # Load model weights
    load_weights(model, get_weights_path(model_name))
    
    # Compute the GFLOPs of the model
    x = tf.constant(np.random.randn(1, 512, 512, 1))
    print(f'GFLOPs: {get_flops(model, [x])}')

    path = args.data_path
    
    # Test variant selection
    if args.test_variant == "internal":
        if model_name in ["UNET_RES18", "FPN_RES18", "FPN_EF0"]:
            eval_test_results(model, model_name, path, RGB=True)
        else:
            eval_test_results(model, model_name, path, RGB=False)
    elif args.test_variant == "external":
        if model_name in ["UNET_RES18", "FPN_RES18", "FPN_EF0"]:
            result, ids = test_model(model, path, RGB=True)
            eval_results(result, ids, model_name)
        else:
            result, ids = test_model(model, path, RGB=False)
            eval_results(result, ids, model_name)
    else:
        raise ValueError(f"Test variant {args.test_variant} is not recognized.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model evaluation script with customizable arguments.")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model to use.", choices=list(weights_mapping.keys()))
    parser.add_argument("--data_path", type=str, required=True, help="Path to test images.")
    parser.add_argument("--test_variant", type=str, choices=["internal", "external"], required=False, default="external", help="Variant of the test set to use: 'internal', 'external'. The INTERNAL test path directory should contain 'JSRT' and 'BSE_JSRT' subdirectories. JSRT contains the original images and BSE_JSRT contains the corresponding ground truth images. The EXTERNAL path directory should contain just the test images.")

    args = parser.parse_args()
    main(args)
