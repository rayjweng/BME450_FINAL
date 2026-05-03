# BME450 Final Project Report

## Project title
Grocery and Beverage Image Sorting for Food Pantry Applications

## Problem statement
This project studies whether a convolutional neural network can separate food and beverage product labels as a first step toward faster pantry item intake. The main task uses the Kaggle image dataset from Tushar Sharma. The supplied grocery dataset from umairaslam is used only for grocery context because it contains basket transactions instead of labeled images.

## Data
1. Image dataset: food and beverage label images from Tushar Sharma
2. Context dataset: grocery basket data from umairaslam
3. Training labels: `labels_(train).csv`
4. Split used in this repo: stratified train, validation, and test split generated from the labeled training images

## Architectures
1. ResNet 18 transfer learning
2. MobileNetV3 Small transfer learning

## Parameter choices
1. Optimizer: Adam, AdamW, SGD
2. Learning rate: 0.01, 0.001, 0.0003
3. Batch size: 32, 64
4. Augmentation: light, heavy
5. Transfer learning scope: pretrained backbone with only the final feature block and classifier trainable

## Results
1. `resnet18_sgd_lr1e2_bs64_heavyaug`
   Architecture: ResNet 18
   Optimizer: SGD
   Learning rate: 0.01
   Batch size: 64
   Augmentation: heavy
   Test accuracy: 0.7462
   Test weighted F1: 0.7457

2. `resnet18_adam_lr1e3_bs32_lightaug`
   Architecture: ResNet 18
   Optimizer: Adam
   Learning rate: 0.001
   Batch size: 32
   Augmentation: light
   Test accuracy: 0.7339
   Test weighted F1: 0.7287

3. `mobilenet_v3_small_adam_lr1e3_bs32_lightaug`
   Architecture: MobileNetV3 Small
   Optimizer: Adam
   Learning rate: 0.001
   Batch size: 32
   Augmentation: light
   Test accuracy: 0.7309
   Test weighted F1: 0.7301

4. `mobilenet_v3_small_adamw_lr3e4_bs64_heavyaug`
   Architecture: MobileNetV3 Small
   Optimizer: AdamW
   Learning rate: 0.0003
   Batch size: 64
   Augmentation: heavy
   Test accuracy: 0.7003
   Test weighted F1: 0.6915

5. `resnet18_adamw_lr3e4_bs32_lightaug`
   Architecture: ResNet 18
   Optimizer: AdamW
   Learning rate: 0.0003
   Batch size: 32
   Augmentation: light
   Test accuracy: 0.6881
   Test weighted F1: 0.6847

## Discussion
1. Compare the two model types.
2. Explain which parameter settings helped or hurt.
3. Note that this is a binary label classification task that stands in for the larger pantry sorting idea.
4. Mention that the first supplied Kaggle link is not an image dataset.
5. In the current run, the best setup was `resnet18_sgd_lr1e2_bs64_heavyaug`, which suggests that stronger augmentation and SGD gave slightly better generalization on this dataset.

## Conclusion
State which experiment performed best and how this pipeline could be extended to the original nine category pantry taxonomy with a larger labeled image set.
