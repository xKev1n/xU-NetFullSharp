## Official repository of the paper xU-NetFullSharp: The novel deep learning architecture for chest X-ray bone shadow suppression

# Introduction
In this paper, an automated deep learning-based framework for bone shadow suppression from frontal CXRs is developed. The framework was inspired by U-Net-based convolutional neural networks (CNNs). 
Among those, a novel neural network architecture called xU-NetFullSharp is proposed. This network is inspired by the most modern U-NetSharp [3] architecture and combines different approaches to preserve as many details, as possible and accurately suppress bone shadows. 
Additionally, recent state-of-the-art CNN models from [4] and [1] designed for this task were used for comparison. Utilized models are available in the `models` folder in the cloud storage.
<p align="center">
  <img src="https://github.com/xKev1n/xU-NetFullSharp/blob/main/images/models/xU-NetFS_EN.svg?raw=true">
</p>

<p align="center">
  <img src="https://github.com/xKev1n/xU-NetFullSharp/blob/main/images/models/DilatedBlockEN.svg?raw=true">
</p>

# Datasets
The experiments utilized two datasets – extensively augmented JSRT and VinDr-CXR [2]. Both datasets are available in the `datasets` folder in the cloud storage.
Firstly, the JSRT dataset containing bone shadow-suppressed CXRs was split into training, validation, and testing sets and was extensively augmented to achieve a sufficient amount of usable images and to ensure the model’s robustness (both original and augmented images are available in the `JSRT` subfolder).
The second, VinDr-CXR, dataset was augmented by randomly applying inversion and used for independent testing (the used testing set is available in the `VinDrCXR` subfolder). 

# Results
Results of the internal testing (on the JSRT dataset) are available in the `internal_test` folder; external testing results (on the VinDr-CXR dataset) are available in the `external_test` folder.
To reproduce the results, use the `test.py` file with desired model and path to corresponding weights.

# References
[1]	S. Kalisz and M. Marczyk, ‘Autoencoder-based bone removal algorithm from x-ray images of the lung’, in 2021 IEEE 21st International Conference on Bioinformatics and Bioengineering (BIBE), 2021, pp. 1–6.

[2]	H. Q. Nguyen et al., ‘VinDr-CXR: An open dataset of chest X-rays with radiologist’s annotations’, 2022.

[3]	L. Qian, X. Zhou, Y. Li, and Z. Hu, ‘UNet#: A UNet-like Redesigning Skip Connections for Medical Image Segmentation’, arXiv preprint arXiv:2205.11759, 2022.

[4]	S. Rajaraman, G. Cohen, L. Spear, L. Folio, and S. Antani, ‘DeBoNet: A deep bone suppression model ensemble to improve disease detection in chest radiographs’, PLoS One, vol. 17, no. 3, p. e0265691, 2022.
