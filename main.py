from utils import check_capabilities, load_model, load_weights, compile_model, train_model, save_model, save_weights, train_debonet, train_kalisz

from UNet import UNet
from Att_UNet import Att_UNet
from UNet3P import UNet3P
from UNetPP import UNetPP
from Att_UNetPP import Att_UNetPP
from DeepResUNet import DeepResUNet
from Att_UNet import Att_UNet
from UNetSharp import UNetSharp
from Kalisz_AE import KaliszAE
from xUNetFS  import xUNetFS
from Att_xUNetFS import Att_xUNetFS

from DeBoNet import DeBoNet

import os
import tensorflow as tf

def main():
    ## Check the availability of GPU
    check_capabilities()

    ## Define the model
    model = xUNetFS()
    
    ## Load weights if needed
    #load_weights(model, "PATH_TO_WEIGHTS")

    # Compile
    model = compile_model(model)

    
    # Training
    #history = train_debonet(model)     ## For DeBoNet
    history = train_model(model)
    #history = train_kalisz(model)      ## For Kalisz Marczyk Autoencoder

if __name__ == "__main__":
    main()
