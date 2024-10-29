# DS-5690-Monosemanticity

paper link: https://transformer-circuits.pub/2024/scaling-monosemanticity/
Author:  Adly Templeton, Tom Conerly, Anthropic

![image](https://github.com/user-attachments/assets/7b719866-717e-456f-9ae8-1ec253bf298a)


## Sparse Autoencoders / SAE
![image](https://github.com/user-attachments/assets/18575db3-c96b-49b3-98e0-37375bc3e159)


## Activation level example
![image](https://github.com/user-attachments/assets/5f03faeb-70b1-4bef-86b4-d4bec66d7c15)


## Different feature scale
![image](https://github.com/user-attachments/assets/391067bd-856f-4569-82dd-64fab0bb78b0)



## False code issue
![image](https://github.com/user-attachments/assets/e3eaf57a-2786-467a-8e68-4d78f8b2d61a)

## Different code activate different feature
![image](https://github.com/user-attachments/assets/15a45ef2-814c-4370-b064-0314271ab501)

## Sparse Autoencoder was applied on residual stream not studying MLP, why?

![image](https://github.com/user-attachments/assets/a50b1c26-c6ec-498a-a5ce-0ba4e3a88264)

## How should we employe this? Espeically on controlling output.

![image](https://github.com/user-attachments/assets/810fabd9-2ff5-4210-8dd9-e789bf181d11)

# Architecture  overview: Pesudo Code section

### Define constants and parameters
D = Dimension of residual stream
F = Number of features

### Input: x (residual stream vector, normalized to unit L2 norm)
### Output: x_hat (reconstructed vector using the learned features)

### Initialize learned parameters
W_enc = encoder weights of size (F x D)
b_enc = encoder biases of size (F,)
W_dec = decoder weights of size (D x F)
b_dec = decoder biases of size (D,)

### Step 1: Compute feature activations f_i(x) for each feature i
for each feature i in range(F):
    # Compute activation using ReLU on the encoder output
    f_i[x] = ReLU(dot(W_enc[i], x) + b_enc[i])

### Step 2: Reconstruct the vector x_hat using the feature activations
x_hat = b_dec
for each feature i in range(F):
    x_hat += f_i[x] * W_dec[:, i]   # Sum the feature contributions weighted by decoder weights

### Step 3: Compute the loss L
### L2 penalty on reconstruction error
reconstruction_loss = ||x - x_hat||^2

### L1 penalty on feature activations (scaled by decoder norms)
feature_penalty = 0
for each feature i in range(F):
    feature_penalty += lambda * f_i[x] * ||W_dec[:, i]||  # Weighted by norm of decoder weights

### Total loss
L = reconstruction_loss + feature_penalty

# Critical Analysis: What should it do further?
1. Propuse ways for unsupervised learning/manipulation.
2. Extracting abstract feature

# Impacts: Superbe

# Related work:
Entropic Activation Steering, https://ar5iv.labs.arxiv.org/html/2406.00244
Contrastive Activation Analysis, https://ar5iv.labs.arxiv.org/html/2406.00045
SPARSE AUTOENCODERS, https://arxiv.org/pdf/2309.08600
DISCOVERING LATENT KNOWLEDGE, https://arxiv.org/pdf/2212.03827
Feature-Based Representations and Shapley Additive Explanations, https://arxiv.org/abs/2409.07132
