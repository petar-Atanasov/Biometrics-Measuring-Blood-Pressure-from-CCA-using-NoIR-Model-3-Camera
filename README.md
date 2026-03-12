# Biometrics: Measuring Blood Pressure from the Common Carotid Artery (CCA) using a Near-Infrared (NoIR) Camera

## Project Overview
This project investigates the feasibility of using **Near-Infrared (NoIR) imaging and machine learning techniques to segment the Common Carotid Artery (CCA)** for potential **biometric authentication and physiological monitorin applications**, particularly in **airport security environments**.

The system aims to demonstrate that **non-invasive vascular imaging** can be used to detect blood flow patterns and potentially estimate **blood pressure and cardiovascular signals** from the CCA.

Unlike traditional biometric systems (fingerprints, facial recognition, iris scans), physiological biometrics susch as **arterial blood flow patterns are significantly harder to spoof**, making them promising for **continous authentification systems**.

The project implements and compares three segmentation techniques: 
- K-Means Clustering
- Convolutional Autoencoder
- UNet++ Deep Learning Architecture

#### Experimental results shows that **UNet++ provides the most anatomically accurate segmentation**, while the other approaches offer lightweight alternatives for embedded systems.
---
## Project Goals
The main objectives of this research are:
- Investigating **NoIR imaging for detecting vascular structures**
- Develop **image preprocessing pipelines for artery enhancement**
- Compare multiple **segmentation algorithms**
- Evaluate segmentation quality using **qualitative and quantative metrics**
- Explore the feasibility of **biometric blood flow monitoring**
---
## System Architecure
The system follows a strucutred computer vision pipeline.
```
NoIR Image Acquisition
        ↓
Image Preprocessing
        ↓
Segmentation Algorithms
        ↓
Evaluation & Analysis
        ↓
Biometric Interpretation
```
#### Images are captured using a **Raspberry Pi NoIR camera**, which detects near-infrared wavelenghts and highlights subdermal blood vessels.
---
## Hardware Setup
The prototype system uses: 
- Raspberry Pi 5
- Raspberry Pi Model 3 NoIR Camera
- Near-infrared imaging pipeline
- Python development environment

#### The camera captures **grayscale NIR images**, allowing visualisation of vascular structures beneath the skin surface.
---
## Image Preprocessing Pipeline
Medical imaging segmentation strongly depends on preprocessing. Several enhancement techniques were applied before segmentation. 

### Contrast Limited Adaptive Histogram Equalization (CLAHE)
CLAHE enhances contrast locally rather than globally.

Benefits:
- Improves vascular visibility
- Handles uneven illumination
- Avoids over-amplification of noise

#### This is critical for low-contrast NoIR images.
---
### Gaussian Smoothing
Gaussian filtering is used to reduce noise in the captured images.

Benefits: 
- Removes high-frequency noise
- Preserves vessel edges
- Stabilizes segmentation algorithms
--- 
### Frangi Vessel Enhancement Filter

#### The Frangi filter highlights tubular structures such as blood vessels.
#### It works by analising the Hessian matrix eigenvalues to detect vessel-like shapes.
#### This step significantly improves artery detection before segmentation.
---
### Region of Interest Extraction (GrabCut)
GrabCut is used to isolate the **neck region where the CCA is located**.

Benefits: 
- Removes background noise
- Focuses segmentation models on relevand anatomy
---
### Edge and Gradient Enhancement 
Additional enhancement techniques include:
- Sobel edge detection
- Gradient-based augmentation
- Red-Channel normalisation
#### These steps improve vascular boundary detection under low light conditions.
---
## Segmentation Algorithms
#### Three different segmentation approaches were implemented and compared.
---
### 1. K-Means Clustering
K-Means is an unsupervised clustering algorithm used as the **baseline segmentation method**.
#### How it works
Pixels are grouped into clusters based on intesity similarity.
```
Minimise:
Sum of squared distances between pixels and cluster centers
```
#### Characterisitics

Advantages:
- Simple
- Fast
- No labelled data required

Limitations:
- No spatial awareness
- Fragmented vessel segmentation
  
#### Optimal performance was achived with **k = 2 clusters**.
---
#### Segmentation Example of K-Means Clustering
<img width="1244" height="331" alt="K-Means-1" src="https://github.com/user-attachments/assets/8c5deb31-637c-4eb4-9f51-ec78a9e4913b" />

---

<img width="1236" height="326" alt="K-Means-3" src="https://github.com/user-attachments/assets/faac2cce-3103-4dfc-ab82-54cd8a615a21" />
K-Means Image Segmentation, a) Enhanced Denoised Image, b) Frangi
Filtered Artery Mask, c) Segmented Ward Linkage Image with K-Means

---

### 2. Covolutional Autoencoder
The second method uses a **deep unsupervised neural network**.
#### Architecure
Encoder
```
  Image
    ↓
Convolution
    ↓
Feature compression
    ↓
Latent representation
```

Decoder
```
Latent Representation
    ↓
Upsampling
    ↓
Reconstructed segmentatiom mask
```

#### The autoencoder learns to reconstruct vascular strucutres from compressed representations. 
---
#### Training
The model was trained for:
```
100 epochs
```
Loss funciton: 
```
Binary Cross Entropy (BCE)
```
Training loss decreased from approximately:
```
0.65 -> 0.30
```
#### showing strong learning convergence. 
--- 

#### Segmentation Example of Convolutional Autoencoder

<img width="1549" height="398" alt="Autoencoder" src="https://github.com/user-attachments/assets/6bd0642a-6dbb-45e4-b862-d25e3c710b22" />

Convolutional Autoencoder Image Segmentation a) Original Image, b)
Autoencoder Feature Enhanced Artery Image, c) Segmentation Red Overlay

### 3. UNet++ Segmentation 
UNet++ is an advanced deep learning architecture designed specifically for **medical image segmentation**.

It extends the original U-Net with:
- nested skip connections
- dense feature propagation
- multi-scale feature extraction

#### These features allow the model to capture fine vascular details.
---

#### Training 
Training configuration:
```
Epochs: 50
Batch size: 8
Loss function:
Binary Cross Entropy + Dice Loss
```

Training loss decreased from: 
```
1.8 -> 1.0
```

Validation loss stabilised around: 
```
~1.6
```
#### indicating good convergence with minor overfitting.
---
#### Segmentation Example of UNet++

<img width="945" height="467" alt="unet++" src="https://github.com/user-attachments/assets/c2af923f-0483-4349-93f8-df71aeba43a0" />

---

## Evaluation Methodology 
The models were evaluated using both **qualitative and quantitative methods**.
---

## Qualitative Evaluation 
Segmentation outputs were visually inspected by overlaying segmentation masks on the original images. 

Evaluation criteria:

- anatomical accuracy
- vessel continuity
- boundary precision 

#### UNet++ showed the best ability to preserve arterial structures even under challenging imaging conditions.
---

## Quantitative Evaluation 
Different metrics were used depending on the model type. 
---
### Silhouette Analysis (K-Means)
Silhouette score measures clustering quality. 

``` 
Score range: -1 to 1
```

Results: 
--- 

<img width="935" height="556" alt="Screenshot 2025-03-29 155858" src="https://github.com/user-attachments/assets/3ff1991a-6a3d-464b-82e8-f83a3feec0de" />

#### Higher values indicate strong cluster separation. 
---

### Training loss Analysis
For deep learning models, performance was evaluated using loss curves.

### Autoencoder: 
---

<img width="973" height="481" alt="Screenshot 2025-03-31 142017" src="https://github.com/user-attachments/assets/1c71ec14-b3bd-47b9-89e6-975adae131e3" />

---
### UNet++
---

<img width="873" height="436" alt="final plot model2 trained (2)" src="https://github.com/user-attachments/assets/95d4747e-45cb-4721-a032-48d1f8f041b2" />

---
#### These results demonstrate stable model learning and convergence. 
---

## Key Results
Main findings from the experiments:
| **Model** | **Strenghts** | **Limitations** |
|------|------|------|
| K- Means | Fast, simple clustering | Fragmented segmentation |
| Autoencoder | Good unsupervised learning | Occasional anatomical errors |
| UNet++ | Highest segmentation accuracy | Higher computational cost |

#### UNet++ produced the most anatomically consistent segmentation results.
---
## Technologies Used
Programming
- Python

Libraries and frameworks
- TensorFlow / Keras
- OpenCV
- NumPy
- Scikit-learn

Hardware
- Raspberry Pi 5
- Raspberry Pi NoIR Camera
