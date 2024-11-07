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

available_models = ["KALISZ_AE", "UNET3P", "UNETPP", "UNET", "ATT_UNET", "ATT_UNETPP", "DEEP_RESUNET", "UNET_SHARP", "XUNETFS", "ATT_XUNETFS", "UNET_RES18", "FPN_RES18", "FPN_EF0"]

def main(args):
    # Model selection based on input argument
    
    model_name = args.model_name
    if model_name == "UNET_RES18":
        model = DeBoNet(COMPILE=False, NAME=model_name)
        # Load model weights
        load_weights(model, os.path.join("weights", model_name, model_name+"_b10_best_weights_11.hdf5"))
    if model_name == "FPN_RES18":
        model = DeBoNet(COMPILE=False, NAME=model_name)
        # Load model weights
        load_weights(model, os.path.join("weights", model_name, model_name+"_b10_best_weights_17.hdf5"))
    if model_name == "FPN_EF0":
        model = DeBoNet(COMPILE=False, NAME=model_name)
        # Load model weights
        load_weights(model, os.path.join("weights", model_name, model_name+"_b10_best_weights_05.hdf5"))
    elif model_name == "KALISZ_AE":
        model = KaliszAE()
        # Load model weights
        load_weights(model, os.path.join("weights", model_name, model_name+"_b10_best_weights_298.hdf5"))
    elif model_name == "UNET3P":
        model = UNet3P()
        # Load model weights
        load_weights(model, os.path.join("weights", model_name, model_name+"_b10_best_weights_37.hdf5"))
    elif model_name == "UNETPP":
        model = UNetPP()
        # Load model weights
        load_weights(model, os.path.join("weights", model_name, model_name+"_b10_best_weights_62.hdf5"))
    elif model_name == "UNET":
        model = UNet()
        # Load model weights
        load_weights(model, os.path.join("weights", model_name, model_name+"_b10_best_weights_82.hdf5"))
    elif model_name == "ATT_UNET":
        model = Att_UNet()
        # Load model weights
        load_weights(model, os.path.join("weights", model_name, model_name+"_b10_best_weights_61.hdf5"))
    elif model_name == "ATT_UNETPP":
        model = Att_UNetPP()
        # Load model weights
        load_weights(model, os.path.join("weights", model_name, model_name+"_b10_best_weights_70.hdf5"))
    elif model_name == "DEEP_RESUNET":
        model = DeepResUNet()
        # Load model weights
        load_weights(model, os.path.join("weights", model_name, model_name+"_b10_best_weights_26.hdf5"))
    elif model_name == "UNET_SHARP":
        model = UNetSharp()
        # Load model weights
        load_weights(model, os.path.join("weights", model_name, model_name+"_b10_best_weights_74.hdf5"))
    elif model_name == "XUNETFS":
        model = xUNetFS()
        # Load model weights
        load_weights(model, os.path.join("weights", model_name, model_name+"_b10_best_weights_30.hdf5"))
    elif model_name == "ATT_XUNETFS":
        model = Att_xUNetFS()
        # Load model weights
        load_weights(model, os.path.join("weights", model_name, model_name+"_b10_best_weights_26.hdf5"))
    else:
        raise ValueError(f"Model {model_name} is not recognized.\nAvailable models: {available_models}")

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
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model to use.", choices=available_models)
    parser.add_argument("--data_path", type=str, required=True, help="Path to test images.")
    parser.add_argument("--test_variant", type=str, choices=["internal", "external"], required=False, default="both", help="Variant of the test set to use: 'internal', 'external'. The INTERNAL test path directory should contain 'JSRT' and 'BSE_JSRT' subdirectories. JSRT contains the original images and BSE_JSRT contains the corresponding ground truth images. The EXTERNAL path directory should contain just the test images.")

    args = parser.parse_args()
    main(args)
