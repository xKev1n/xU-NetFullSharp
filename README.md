## Official repository of the paper xU-NetFullSharp: The novel deep learning architecture for chest X-ray bone shadow suppression

In this paper, an automated deep learning-based framework for bone shadow suppression from frontal CXRs is developed. The framework was inspired by U-Net-based convolutional neural networks (CNNs). 
Among those, novel architecture of a neural network, called xU-NetFullSharp, is proposed. This network is inspired by the most modern U-NetSharp architecture and combines different approaches to preserve as many details, as possible and accurately suppress bone shadows. 
Additionally, recent state-of-the-art CNN models designed for this task were used for comparison. Utilized models are available in the `models` folder in the cloud storage.
The experiments utilized two datasets – extensively augmented JSRT and VinDr-CXR. Both datasets are available in the `datasets` folder in the cloud storage.
Firstly, the JSRT dataset containing bone shadow-suppressed CXRs was split into training, validation, and testing sets and was extensively augmented to achieve a sufficient amount of usable images and to ensure the model’s robustness (both original and augmented images are available in the `JSRT` subfolder).
Second, VinDr-CXR, dataset was augmented by randomly applying inversion and used for independent testing (the used testing set is available in the `VinDrCXR` subfolder). 
The results of internal testing (on the JSRT dataset) are available in the `internal_test` folder; external testing results (on the VinDr-CXR dataset) are available in the `external_test` folder.
