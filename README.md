# Skin Cancer Detection using Deep Learning

## Introduction
Skin cancer is one of the most common cancers worldwide, and early detection is critical for effective treatment. Traditional diagnostic methods rely on visual examination and biopsy, which can be time-consuming and require specialist expertise. Recent advancements in **deep learning** have shown promise in automating and improving the accuracy of skin cancer detection. In this project, we explore the application of **convolutional neural networks (CNNs)** and **transfer learning** using **EfficientNetB7** to classify different types of skin cancer from dermoscopic images. This research is particularly interesting as it has the potential to assist dermatologists and improve patient outcomes through faster and more accurate diagnoses.

## Methods

### Dataset Description
- The dataset used in this study is from the **International Skin Imaging Collaboration (ISIC)**, obtained from **Kaggle**. It consists of dermoscopic images categorized into different types of skin lesions, including melanoma, basal cell carcinoma, and others.
- Number of classes: 9

### Data Loading and Preprocessing
- The dataset was organized into 'Test' and 'Train' folders, with subfolders for different classes of skin lesions. Image paths were captured in a Python dictionary for processing.
- Addressed class imbalance by sampling images to ensure each class had an equal number of images.
- Images were resized to **224x224 pixels** and normalized. **Data augmentation** techniques were applied to training images to increase variability and prevent overfitting.
- Converted image arrays into **tensors** and prepared data for model training and testing. Labels were **one-hot encoded** for model compatibility.

### Model Development
#### Custom CNN Model
- Developed a **custom CNN model** with **pooling layers**, **convolutional layers**, **dense layers**, and a **flatten layer**. **Dropout layers** were used to prevent overfitting. **ReLU** and **softmax** activation functions were used, with **categorical cross-entropy** as the loss function.
- Training was conducted over **10 epochs** using the **Adam optimizer** with a learning rate of **0.001**.

#### Transfer Learning with EfficientNetB7
- Implemented **transfer learning** using **EfficientNetB7**, a **pre-trained model** on **ImageNet**. EfficientNetB7 was chosen for its efficiency and performance.
- Frozen initial layers to retain pre-trained weights and added **custom layers** for lesion classification. Fine-tuned top layers with a lower learning rate.
- Used **Adam optimizer** with an initial learning rate of **0.001**, and **categorical cross-entropy** loss function. **Accuracy** was the primary evaluation metric.

## Results

- **Custom CNN Model:**
  - Accuracy after 10 epochs: **48.23%**
  - Confusion matrix indicated misclassification of all images as “pigment benign keratosis.”
  
- **EfficientNetB7 Model:**
  - Accuracy after 5 epochs: **14.03%**
  - Confusion matrix indicated similar issues as the custom model.

## Discussion
- Both models struggled with classification accuracy, indicating potential issues with model architecture, hyperparameters, or data preprocessing.
- Suggested steps to improve performance include:
  - Testing for **preprocessing errors**
  - Experimenting with different **model architectures** and **regularization techniques**
  - Ensuring consistent **data preprocessing**
  - Utilizing **GPU access** for faster model training
  - Applying **cross-validation** and **early stopping** to optimize hyperparameters
