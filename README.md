# Adversarial Domain Adaptation

Following is a **_Keras_** implementation of an Adversarial Domain Adaptation Model that assigns class labels to images in the Target domain by extracting domain-invariant features from the labelled Source and unlabelled Target domain samples. The architecture involves three sub-networks: _(a) domain-invariant feature extractor, (b) label classifier and (c) domain discriminator._


The following code has drawn inspiration from the following papers:

- *Ganin, Yaroslav, and Victor Lempitsky. "Unsupervised domain adaptation by backpropagation." arXiv preprint arXiv:1409.7495 (2014)*.

- *Li, Yanghao, Naiyan Wang, Jianping Shi, Jiaying Liu, and Xiaodi Hou. "Revisiting batch normalization for practical domain adaptation." arXiv preprint arXiv:1603.04779 (2016)*.

- *Tzeng, Eric, Judy Hoffman, Kate Saenko, and Trevor Darrell. "Adversarial discriminative domain adaptation." In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 7167-7176. 2017*.

**Dataset Download**

The code is tested on the **_Office-31_** dataset. 
> Download it from this link: *https://drive.google.com/file/d/0B4IapRTv9pJ1WGZVd1VDMmhwdlE/view*.\
> Create a *domain_adaptation_images* directory under *Data/Office/* and place the downloaded images inside.

**Requirements**

The code is compatible with the mentioned versions of the following libraries. However, it might also work with their previous versions.

- pillow 6.0.0
- scikit-learn 0.20.2
- tensorflow 1.12.0
- keras 2.2.4

**Model Run**

The model outputs the _Source_ and _Target classification accuracies_. 

- An example running code is given below:

> *python driver.py --batch_size 32 --number_of_gpus 2 --lr_combined 0.00001 --num_iterations 5000*

- Acceptable Parameters:

> *--number_of_gpus*, default = '1' : *"Number of gpus required to run"*\
> *--network_name*, default = 'ResNet50' : *"Name of the feature extractor network"*\
> *--dataset_name*, default = 'Office' : *"Name of the source dataset"*\
> *--dropout_classifier*, default = 0.25 : *"Dropout ratio for classifier"*\
> *--dropout_discriminator*, default = 0.25 : *"Dropout ratio for discriminator"*\
> *--source_path*, default = 'amazon_10_list.txt' : *"Path to source dataset"*\
> *--target_path*, default = 'webcam_10_list.txt' : *"Path to target dataset"*\
> *--lr_classifier*, default = 0.0001 : *"Learning rate for classifier model"*\
> *--lr_discriminator*, default = 0.0001 : *"Learning rate for discriminator model"*\
> *--lr_combined*, default = 0.00001 : *"Learning rate for combined model"*\
> *--classifier_loss_weight*, default = 1 : *"Classifier loss weight"*\
> *--discriminator_loss_weight*, default = 2 : *"Discriminator loss weight"*\
> *--batch_size*, default = 16 : *"Batch size for training"*\
> *--test_interval*, default = 3 : *"Gap between two successive test phases"*\
> *--num_iterations*, default = 1000 : *"Number of iterations"*\
> *--snapshot_interval*, default = 100 : *"Gap between saving output models"*\
> *--output_dir*, default = 'models' : *"Directory for saving output model"*\
