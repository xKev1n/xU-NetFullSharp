import argparse
from utils import train_model, load_model, load_weights, compile_model, train_debonet, train_kalisz
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

available_models = ["KALISZ_AE", "UNET3P", "UNETPP", "UNET", "ATT_UNET", "ATT_UNETPP", "DEEP_RESUNET", "UNET_SHARP", "XUNETFS", "ATT_XUNETFS", "UNET_RES18", "FPN_RES18", "FPN_EF0"]

def main(args):
    # Model selection based on input argument
    
    model_name = args.model_name
    if model_name == "UNET_RES18":
        model = DeBoNet(COMPILE=False, NAME=model_name)
        # Load model weights if needed
        if args.weights_path:
            load_weights(model, args.weights_path)
    elif model_name == "FPN_RES18":
        model = DeBoNet(COMPILE=False, NAME=model_name)
        # Load model weights if needed
        if args.weights_path:
            load_weights(model, args.weights_path)
    elif model_name == "FPN_EF0":
        model = DeBoNet(COMPILE=False, NAME=model_name)
        # Load model weights if needed
        if args.weights_path:
            load_weights(model, args.weights_path)
    elif model_name == "KALISZ_AE":
        model = KaliszAE()
        # Load model weights if needed
        if args.weights_path:
            load_weights(model, args.weights_path)
    elif model_name == "UNET3P":
        model = UNet3P()
        # Load model weights if needed
        if args.weights_path:
            load_weights(model, args.weights_path)
    elif model_name == "UNETPP":
        model = UNetPP()
        # Load model weights if needed
        if args.weights_path:
            load_weights(model, args.weights_path)
    elif model_name == "UNET":
        model = UNet()
        # Load model weights if needed
        if args.weights_path:
            load_weights(model, args.weights_path)
    elif model_name == "ATT_UNET":
        model = Att_UNet()
        # Load model weights if needed
        if args.weights_path:
            load_weights(model, args.weights_path)
    elif model_name == "ATT_UNETPP":
        model = Att_UNetPP()
        # Load model weights if needed
        if args.weights_path:
            load_weights(model, args.weights_path)
    elif model_name == "DEEP_RESUNET":
        model = DeepResUNet()
        # Load model weights if needed
        if args.weights_path:
            load_weights(model, args.weights_path)
    elif model_name == "UNET_SHARP":
        model = UNetSharp()
        # Load model weights if needed
        if args.weights_path:
            load_weights(model, args.weights_path)
    elif model_name == "XUNETFS":
        model = xUNetFS()
        # Load model weights if needed
        if args.weights_path:
            load_weights(model, args.weights_path)
    elif model_name == "ATT_XUNETFS":
        model = Att_xUNetFS()
        # Load model weights if needed
        if args.weights_path:
            load_weights(model, args.weights_path)
    else:
        raise ValueError(f"Model {model_name} is not recognized.\nAvailable models: {available_models}")

    # Compile the model
    model = compile_model(model)

    # Training
    if model_name == "KALISZ_AE":
        history = train_kalisz(model, args.data_path, model_name)
    elif model_name in ["UNET_RES18", "FPN_RES18", "FPN_EF0"]:
        history = train_debonet(model, args.data_path, model_name)
    else:
        history = train_model(model, args.data_path, model_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model training script with customizable arguments.")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model to use.", choices=available_models)
    parser.add_argument("--data_path", type=str, required=True, help="Path to training data. The training directory should contain 'train' and 'val' subdirectories. The 'train' and 'val' directories should contain 'JSRT' and 'BSE_JSRT' subdirectories. JSRT contains the original images and BSE_JSRT contains the corresponding ground truth images.")
    parser.add_argument("--weights_path", type=str, required=False, help="Path to model weights to load before training.")

    args = parser.parse_args()
    main(args)