from utils import eval_results, test_model, load_model, load_weights, compile_model, eval_test_results, eval_test_results_woPred
from UNet3P import UNet3P
from UNetPP import UNetPP
from UNet import UNet
from Att_UNet import Att_UNet
from Att_UNetPP import Att_UNetPP
from DeepResUNet import DeepResUNet
from UNetSharp import UNetSharp
from DeBoNet import DeBoNet
from Kalisz_AE import KaliszAE
from xUNetFS import xUNetFS
from Att_xUNetFS import Att_xUNetFS

def main():
    # Define the name of the model
    model_name='ATT_UNET'
    
    #model = DeBoNet(COMPILE=False, NAME=model_name)        # If DeBoNet
    model = xUNetFS(DS=True)
    
    load_weights(model, ".tf_checkpoints/512/"+model_name+"/"+model_name+"_b10_f256_best_weights_61.hdf5")

    ## INTERNAL TEST SET
    eval_test_results(model, model_name, RGB=False)         # RGB = True is necessary for DeBoNet models
    #eval_test_results_woPred("predictions/512/DEBONET/", "ribs_suppresion/new/augmented/test/BSE_JSRT/", model_name)       # If DeBoNet
    
    ## TESTING UNSEEN IMAGES -- EXTERNAL TEST SET
    result, ids = test_model(model)
    eval_results(result, ids, model_name)
    
if __name__ == "__main__":
    main()