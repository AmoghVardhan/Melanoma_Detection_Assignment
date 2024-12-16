# Melanoma Skin Cancer Detection

## Abstract
Melanoma is the deadliest form of skin cancer among the over 200 different cancer types. The diagnostic process for melanoma begins with clinical screening, followed by dermoscopic analysis and histopathological examination. Early detection of melanoma significantly improves the chances of successful treatment. The initial step involves a visual inspection of the affected skin area, where dermatologists use dermatoscopic images captured by high-speed cameras. These images, when analyzed visually, achieve an accuracy of 65-80% in melanoma diagnosis. Combining dermatoscopic images with further examination by specialists raises the prediction accuracy to 75-84%. This project aims to build an automated classification system leveraging image processing techniques to classify skin cancer using skin lesion images.

## Problem Statement
The traditional diagnostic process for melanoma involves a skin biopsy, where a dermatologist examines a sample of the lesion under a microscope. This process, from scheduling an appointment to receiving biopsy results, can take over a week. This project aims to reduce this time gap to just a couple of days by providing a predictive model that can automate the classification of skin lesions. The proposed approach uses a Convolutional Neural Network (CNN) to classify multiple types of skin cancer based on lesion images, offering a faster and scalable solution that can positively impact millions of lives.

## Motivation
The primary goal of this project is to contribute to reducing deaths caused by melanoma through early detection. By leveraging advancements in computer vision, machine learning, and deep learning, the project aims to apply state-of-the-art image classification techniques for the well-being of individuals worldwide.

## Dataset
The dataset comprises 2,357 images of malignant and benign oncological diseases, sourced from the International Skin Imaging Collaboration (ISIC). The images are categorized based on their respective classes and divided into balanced subsets to ensure consistent representation across all types.

To address the class imbalance issue, the Augmentor Python package (https://augmentor.readthedocs.io/en/master/) was used to generate additional samples for underrepresented classes.

## CNN Architecture Design
The CNN model was designed to classify skin cancer using lesion images. The architecture prioritizes performance and generalization by including key layers to extract features and prevent overfitting.

### Model Architecture
CNN Model Design
To achieve accurate classification of melanoma and other skin cancers, a custom Convolutional Neural Network (CNN) architecture was implemented. The model incorporates the following layers:

- **Rescaling Layer**: Normalizes input images from the [0, 255] range to [0, 1].
- **Convolutional Layer**: Extracts features from the input by applying convolution operations, reducing image dimensions while retaining key information.
- **Pooling Layer**: Reduces feature map dimensions, decreasing computational complexity and summarizing features.
- **Dropout Layer**: Randomly deactivates neurons during training to prevent overfitting.
- **Flatten Layer**: Converts multidimensional outputs into a single feature vector for the dense layer.
- **Dense Layer**: Connects all neurons to the previous layer for comprehensive learning.
- **Activation Functions**:
  - **ReLU**: Addresses vanishing gradient issues and accelerates learning by outputting zero for negative values.
  - **Softmax**: Converts outputs into probabilities, ensuring the sum of all probabilities equals one.

<img width="787" alt="image" src="https://github.com/user-attachments/assets/0449ae52-e232-43d7-ae36-e8f2b8faf6c4" />


### Key Layers:
- **Rescaling Layer**: Rescales input pixel values from [0, 255] to [0, 1].
- **Convolutional Layers**: Extracts features from input images using filters to reduce spatial dimensions.
- **MaxPooling Layers**: Reduces feature map dimensions, minimizing computational cost and overfitting.
- **Flatten Layer**: Converts feature maps into a 1D array for input into dense layers.
- **Dense Layer**: Fully connected layer for learning complex relationships in features.
- **Dropout Layer**: Randomly drops neurons during training (rate = 0.5) to prevent overfitting.
- **Softmax Activation**: Outputs a probability distribution for multi-class classification.


## Model Evaluation
The model was evaluated on accuracy and loss metrics. The following evaluation metrics were achieved: 
<img width="718" alt="image" src="https://github.com/user-attachments/assets/9a5586f7-4bae-41b4-bd7b-557168b86b89" />

## Findings:

### Training Accuracy and Validation Accuracy:

- The training accuracy reaches approximately 0.86 by the end of training, indicating the model is learning the training data well.
- The validation accuracy stabilizes around 0.83, which shows a good improvement compared to earlier results. However, there is still a slight gap between training and validation accuracy.

### Training Loss and Validation Loss:

- The training loss steadily decreases to approximately 0.40, showing that the model is fitting the training data effectively.
- The validation loss decreases initially but plateaus and fluctuates around 0.75 after epoch 20, indicating that the model may still struggle slightly to generalize on unseen data.

Class Imbalance Fix:

Adding 500 additional images per class using the Augmentor library appears to have significantly improved the overall model performance. The validation accuracy has increased, and overfitting has been reduced compared to previous results.

## References
1. [Melanoma Skin Cancer Overview](https://www.cancer.org/cancer/melanoma-skin-cancer/about/what-is-melanoma.html)
2. [Introduction to CNN](https://www.analyticsvidhya.com/blog/2021/05/convolutional-neural-networks-cnn/)
3. [Image Classification Using CNN](https://www.analyticsvidhya.com/blog/2020/02/learn-image-classification-cnn-convolutional-neural-networks-3-datasets/)
4. [Efficient CNN Design](https://towardsdatascience.com/a-guide-to-an-efficient-way-to-build-neural-network-architectures-part-ii-hyper-parameter-42efca01e5d7)
