# Multi_rate_vae_gdl



# datasets:
(1) Can MR-VAEs learn the optimal response functions for linear VAEs? 
(2) Does our method scale to modern-size β-VAEs? 
(3) How sensitive are MR-VAEs to hyperparameters? 
(4) Can MR-VAEs be applied to other VAE models such as β-TCVAEs (Chen et al., 2018)?


(1) HOW DO MR-VAES PERFORM ON LINEAR VAES? (page 7 section 5.1)
    - two-layer linear VAE models (introduced in Section 3.2) 
    - We trained MR-VAEs on the MNIST dataset (Deng, 2012) by sampling the KL weighting term β from 0.01 to 10.
    - Then, we trained 10 separate two-layer linear β-VAEs, each with different β values. 

(2) CAN MR-VAES SCALE TO MODERN-SIZE ARCHITECTURES? (page 7, section 5.2)
    - We trained convolution and ResNet-based architectures on binary static MNIST, Omniglot, CIFAR-10, SVHN, CelebA64 ollowing the experimental set-up from Chadebec et al. (2022). #  For MNIST and CIFAR10 the validation set is composed of the last 10k images extracted from the official train set and the test set corresponds to the official one,  For CELEBA, we use the official train/val/test split
    - We also trained NVAEs on binary static MNIST, Omniglot, and CelebA datasets using the same experimental set-up from Vahdat & Kautz (2020).
    - Lastly, we trained autoregressive LSTM VAEs on the Yahoo dataset with the set-up from He et al. (2019).
    - We sampled the KL weighting term from 0.01 to 10 for both MR-VAEs and β-VAEs. Note that the training for β-VAEs was repeated 10 times 
      (4 times for NVAEs) with log uniformly spaced β to estimate a rate-distortion curve.
    - Note that traditional VAEs typically require scheduling the KL weight to avoid posterior collapse nd we train β-VAEs with both a constant and KL annealing schedule.
    - Note that we focus on the visualization of the rate-distortion curve as advocated for by Alemi et al. (2016); Huang et al. (2020).
    - We further show samples from both β-VAEs and MR-VAEs on SVHN and CelebA datasets which used ResNet-based encoders and decoders in Figure 7.

(3) HOW SENSITIVE ARE MR-VAES TO HYPERPARAMETERS? (page 8, section 5.3)
    - we show that MR-VAEs are insensitive and robust to the choices of the hyperparameters and can be fixed through various applications. 
    - We trained ResNet encoders and decoders on the Omniglot dataset with the same configuration as Section 5.2 and show the test rate-distortion curves in Figure 8. 
      In the left, we fixed b = 10 and changed the sample range a in the set {0.001, 0.01, 0.1, 1} and in the middle, we fixed a = 0.01 and modified the sample range b in the set {10, 1, 0.1}.


(4) CAN MR-VAES BE EXTENDED TO OTHER MODELS? (page 9, section 5.4)
    - To demonstrate the applicability of MR-VAEs to other VAE models, we trained MR-VAEs on the β- TCVAE objective. The weight in β-TCVAE balances the reconstruction error and the total correlation instead of the reconstruction error and the KL divergence term.
    - We trained MR-VAEs composed of MLP encoders and decoders on the dSprites dataset, following the set-up from Chen et al. (2018). 
    - Since we are interested in the disentanglement ability of the model, we sampled the weighting β between 0.1 and 10.


DATASETS:
    - binary static MNIST, Omniglot, CIFAR-10, SVHN, CelebA64 (or CelebA), Yahoo dataset

EXPERIMENT DETAILS (read appendix C):
    (1) linear vae: - Adam optimizer, 
                    - 200 epochs, 
                    - 10 epochs of learning rate warmup and a cosine learning rate decay
                    - latent dimension was set to 32
                    - hyperparameter searches over learning rates {0.01, 0.003, 0.001, 0.0003, 0.0001, 0.00003, 0.00001}
                    - βs uniformly sampled between 0.01 and 10 on a log scale
                    - We selected the learning rate that achieved the lowest average training loss.
                    - The experiments were repeated 3 times with different random seeds, and the mean is presented.

        mrvae: - same training and architectural configurations (as above)
               - We performed a grid search on learning rates {0.01, 0.003, 0.001, 0.0003, 0.0001, 0.00003, 0.00001} and selected the learning rate that achieved the best training loss at β = 1
               - We also repeated the experiments 3 times with different random seeds.

    (2) IMAGE RECONSTRUCTION: - Adam optimizer for CNN and ResNet 200 epochs with a batch size of 128 and a cosine learning rate decay.
                              - For NVAE architecture with binary images, we used the Adamax optimizer for 400 epochs with a batch size of 256 and a cosine learning rate decay.  
                              - CelebA64 dataset, we use a batch size of 16.


WHERE TO FIND DATASET: (torchvision.datasets)
    1. MNIST on pytorch 
    2. Omniglot on pytorch 
    3. CIFAR-10 on pytorch 
    4. SVHN on pytorch 
    5. CelebA on pytorch 
    6. Yahoo no idea (found something on pytorch: https://pytorch.org/text/stable/_modules/torchtext/datasets/yahooanswers.html)





