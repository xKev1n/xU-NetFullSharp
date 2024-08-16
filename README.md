## Official repository of the paper xU-NetFullSharp: The novel deep learning architecture for chest X-ray bone shadow suppression

# Introduction
In this paper, an automated deep learning-based framework for bone shadow suppression from frontal CXRs is developed. The framework was inspired by U-Net-based convolutional neural networks (CNNs). 
Among those, a novel neural network architecture called xU-NetFullSharp is proposed. This network is inspired by the most modern U-NetSharp [5] architecture and combines different approaches to preserve as many details, as possible and accurately suppress bone shadows. 
Additionally, recent state-of-the-art CNN models from [6] and [2] designed for this task were used for comparison. Utilized models are available in the `models` folder in the cloud storage.

# The proposed architecture
The xU-NetFullSharp is based on the most recent U-NetSharp [5] architecture and utilizes multi-scale skip connections both ways in a similar way to the preceding U-Net3+ [1]. The ReLU activation is changed for more modern xUnit [3] activation to ensure more relevant activation maps.

<p align="center">
  <img src="https://github.com/xKev1n/xU-NetFullSharp/blob/main/images/models/xU-NetFS_EN.svg?raw=true" alt>
</p>
<p align="center">
  <em>The architecture of the proposed xU-NetFullSharp</em>
</p>

Blocks of the proposed architecture are made up of 2D convolutions with different rates of dilation and xUnit [3] activation functions.
<p align="center">
  <img src="https://github.com/xKev1n/xU-NetFullSharp/blob/main/images/models/DilatedBlockEN.svg?raw=true" alt>
</p>
<p align="center">
  <em>The structure of dilated blocks</em>
</p>

# Datasets
The experiments utilized two datasets – extensively augmented JSRT and VinDr-CXR [4]. Both datasets are available in the `datasets` folder in the [cloud storage](https://drive.google.com/file/d/1f0LP05jhNPI0UjqkQhpAyBbp8KV_2y4Y/view?usp=drive_link).
Firstly, the JSRT dataset containing bone shadow-suppressed CXRs was split into training, validation, and testing sets and was extensively augmented to achieve a sufficient amount of usable images and to ensure the model’s robustness (both original and augmented images are available in the `JSRT` subfolder).
The second, VinDr-CXR, dataset was augmented by randomly applying inversion and used for independent testing (the used testing set is available in the `VinDrCXR` subfolder). 

# Results
Results of the internal testing (on the JSRT dataset) are available in the `internal_test` folder; external testing results (on the VinDr-CXR dataset) are available in the `external_test` folder.
To reproduce the results, use the `test.py` file with the desired model and path to corresponding weights. Sample outputs from individual models can be seen in the `/images/xray` folder in this repository.

# Objective results
| **Models**                         | **MAE**    | **MSE**    | **SSIM**   | **MS-SSIM** | **UIQI**   | **PSNR [dB]** |
| ---------------------------------- | ---------- | ---------- | ---------- | ----------- | ---------- | ------------- |
| **U-Net**                          | 0.0074     | 0.0004     | 0.9835     | 0.9865      | 0.9959     | 34.6081       |
| **Attention U-Net**                | 0.0080     | 0.0004     | 0.9834     | 0.9863      | 0.9960     | 34.4984       |
| **Deep Residual U-Net**            | 0.0120     | 0.0007     | 0.9686     | 0.9768      | 0.9903     | 31.5990       |
| **U-Net++**                        | 0.0073     | 0.0004     | 0.9834     | 0.9867      | 0.9959     | 34.6063       |
| **Attention U-Net++**              | 0.00076    | 0.0004     | 0.9835     | 0.9861      | 0.9957     | 34.4042       |
| **U-Net3+**                        | 0.0074     | 0.0004     | 0.9836     | 0.9868      | 0.9957     | 34.6300       |
| **U-NetSharp**                     | 0.0078     | 0.0004     | 0.9836     | 0.9868      | 0.9960     | 34.4829       |
| **xU-NetFullSharp**                | **0.0071** | **0.0003** | **0.9846** | **0.9870**  | **0.9961** | 34.5338       |
| **Attention xU-NetFullSharp**      | 0.0076     | **0.0003** | 0.9841     | 0.9869      | 0.9960     | **34.7016**   |
| **Kalisz Marczyk’s Autoencoder**   | 0.0097     | 0.0005     | 0.9722     | 0.9821      | 0.9948     | 33.3040       |
| **FPN-ResNet-18**                  | 0.0154     | 0.0006     | 0.9708     | 0.9781      | 0.9917     | 31.7906       |
| **FPN-EfficientNet-B0**            | 0.0164     | 0.0006     | 0.9694     | 0.9802      | 0.9917     | 31.6370       |
| **U-Net-ResNet-18**                | 0.0172     | 0.0008     | 0.9660     | 0.9713      | 0.9904     | 30.7114       |
| **DeBoNet**                        | 0.0159     | 0.0022     | 0.9312     | 0.9642      | 0.9953     | 27.0403       |

# Histogram comparison
| **Models**                         | **Correlation** | **Intersection** | **𝜒<sup>2</sup>** | **Bhattacharyya** |
| ---------------------------------- | --------------- | ---------------- | ------------------ | ----------------- |
| **U-Net**                          | 0.9443          | 9.8999           | 6.1702             | 0.1292            |
| **Attention U-Net**                | 0.9449          | 9.8873           | 8.3925             | 0.1295            |
| **Deep Residual U-Net**            | 0.9358          | 9.5917           | 5.8152             | 0.1478            |
| **U-Net++**                        | 0.9394          | 9.8962           | **4.8495**         | 0.1309            |
| **Attention U-Net++**              | 0.9478          | 9.9733           | 12.5015            | 0.1202            |
| **U-Net3+**                        | 0.9329          | 9.9043           | 8.3654             | 0.1251            |
| **U-NetSharp**                     | 0.9414          | 9.9133           | 14.6634            | 0.1330            |
| **Attention xU-NetFullSharp**      | 0.9321          | 9.8601           | 8.9438             | 0.1396            |
| **xU-NetFullSharp**                | **0.9631**      | **10.0285**      | 7.0048             | **0.1155**        |
| **Kalisz Marczyk’s Autoencoder**   | 0.9580          | 9.8830           | 6.7579             | 0.1220            |
| **FPN-ResNet-18**                  | 0.8658          | 9.1395           | 5.1873             | 0.1976            |
| **FPN-EfficientNet-B0**            | 0.8763          | 9.3075           | 12.9367            | 0.1831            |
| **U-Net-ResNet-18**                | 0.8843          | 9.0170           | 26.0124            | 0.1973            |
| **DeBoNet**                        | 0.9273          | 9.6682           | 11.0798            | 0.1418            |

# Experts' rating of the results on the external VinDr-CXR dataset
| **Model**                             | **Average rating**(best = 1)  | **Expert’s comment**                                                              |                   |
| ------------------------------------- | ------------------------------ | --------------------------------------------------------------------------------- | ----------------- |
| **U-Net**                             | 3.0                            |                                                                                   |                   |
| **Attention** **U-Net**               | 2.8                            |                                                                                   |                   |
| **U-Net++**                           | 3.0                            |                                                                                   |                   |
| **Attention** **U-Net++**             | **2.3**                        |                                                                                   |                   |
| **U-Net3+**                           | 2.8                            | **Some problems with detail retention**                                           |                   |
| **U-NetSharp**                        | **1.7**                        | **Second best in bone shadow suppression. Great retention of details.**           |                   |
| **xU-NetFullSharp**                   | **1.2**                        | **Consistently the best in bone shadow suppression. Great retention of details.** |                   |
| **Attention** **xU-NetFullSharp**     | 3.0                            |                                                                                   |                   |
| **U-Net-ResNet-18**                   | 4.5                            |                                                                                   | Concurrent models |
| **FPN-ResNet-18**                     | 4.3                            | **Major deformation of the chest silhouette in one sample!**                      |                   |
| **FPN-EfficientNet-B0**               | 3.7                            | **Blurry details**                                                                |                   |
| **Kalisz Marczyk’s Autoencoder**      | 3.5                            |                                                                                   |                   |

# Experts' rating of the results on the external Gusarev DES dataset
| **Model**                                      | **Vessel visibility**                                                    | **Airway visibility**                                                                 | **Bone shadow suppression**       | **Overall bone shadow suppression performance**            | **Nodule visibility** |
| ---------------------------------------------- | ------------------------------------------------------------------------ | ------------------------------------------------------------------------------------- | --------------------------------- | ---------------------------------------------------------- | --------------------- |
| ---------------------------------------------- | 3: Clearly visible 2: Visible 1: Not visible 	 						| 3: Lobar and intermediate bronchi 2: Main bronchus and rump 1: Trachea 				| 3: Nearly perfect 2: Less than 5 unsuppressed bones 1: 5 or more unsuppressed bones 	| 1: Excellent 3: Average 5: Poor 	| 3: More apparent 2: Equally as apparent 1: Less apparent 	 |
| **DES (Reference)**                            | **3**                                                                    | **1.8**                                                                               | **2.7**                           | **1.3**                                                    | **2.9**               |
| **Attention U-Net**                            | **3**                                                                    | 2.1                                                                                   | **1**                             | 2.8                                                        | **2.3**               |
| **Attention U-Net++**                          | **3**                                                                    | 2.2                                                                                   | **1**                             | 2.3                                                        | 2.1                   |
| **Attention xU-NetFullSharp**                  | **3**                                                                    | 2.3                                                                                   | **1**                             | 2.4                                                        | 2                     |
| **DeBoNet**                                    | **2**                                                                    | 1.8                                                                                   | **1**                             | 4.2                                                        | 1.4                   |
| **FPN-EfficientNet-B0**                        | **3**                                                                    | 2.2                                                                                   | **1**                             | 3.3                                                        | 1.5                   |
| **FPN-ResNet-18**                              | **3**                                                                    | 2.3                                                                                   | **1**                             | 3.5                                                        | 2                     |
| **U-Net-ResNet-18**                            | **3**                                                                    | 2.3                                                                                   | **1**                             | 4.9                                                        | 2.1                   |
| **Deep Residual U-Net**                        | 2.2                                                                      | 2                                                                                     | **1**                             | 5                                                          | 1.9                   |
| **Kalisz Marczyk’s Autoencoder**               | **3**                                                                    | 2.3                                                                                   | **1**                             | 3.1                                                        | 2.1                   |
| **U-Net**                                      | **3**                                                                    | 2.3                                                                                   | **1**                             | 2.6                                                        | 2.2                   |
| **U-NetSharp**                                 | **3**                                                                    | **2.4**                                                                               | **1**                             | 2.1                                                        | 2.2                   |
| **U-Net3+**                                    | **3**                                                                    | **2.4**                                                                               | **1**                             | 2.5                                                        | 2.2                   |
| **U-Net++**                                    | **3**                                                                    | **2.4**                                                                               | **1**                             | 2.2                                                        | 2.1                   |
| **xU-NetFullSharp**                            | **3**                                                                    | **2.4**                                                                               | **1**                             | **1.6**                                                    | **2.3**               |

# References
[1] H. Huang et al., ‘UNet 3+: A Full-Scale Connected UNet for Medical Image Segmentation’, in ICASSP 2020 - 2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2020, pp. 1055–1059. doi: 10.1109/ICASSP40776.2020.9053405.

[2]	S. Kalisz and M. Marczyk, ‘Autoencoder-based bone removal algorithm from x-ray images of the lung’, in 2021 IEEE 21st International Conference on Bioinformatics and Bioengineering (BIBE), 2021, pp. 1–6.

[3] I. Kligvasser, T. R. Shaham, and T. Michaeli, ‘xUnit: Learning a Spatial Activation Function for Efficient Image  Restoration’, CoRR, vol. abs/1711.06445, 2017, [Online]. Available: http://arxiv.org/abs/1711.06445

[4]	H. Q. Nguyen et al., ‘VinDr-CXR: An open dataset of chest X-rays with radiologist’s annotations’, 2022.

[5]	L. Qian, X. Zhou, Y. Li, and Z. Hu, ‘UNet#: A UNet-like Redesigning Skip Connections for Medical Image Segmentation’, arXiv preprint arXiv:2205.11759, 2022.

[6]	S. Rajaraman, G. Cohen, L. Spear, L. Folio, and S. Antani, ‘DeBoNet: A deep bone suppression model ensemble to improve disease detection in chest radiographs’, PLoS One, vol. 17, no. 3, p. e0265691, 2022.
