# Comprehensive Approach to Dogs vs. Cats Classification with Dataset Fusion, Deep Learning Models, and Real-time Prediction Deployment

## Introduction

This comprehensive project reimagines the traditional "Dogs vs. Cats" image classification problem by incorporating a unique twist. The goal is to create a comprehensive dataset that combines two distinct datasets: "Dogs vs. Cats" and "Dog Breed Identification." This innovative approach not only adds complexity to the classification task but also explores the versatility of deep learning models. The project encompasses data collection, label extraction, dataset organization, model training using both TensorFlow and PyTorch, and the deployment of ONNX models for real-time predictions via FastAPI.

## Dataset Fusion

**Step 1: Data Collection**
- We access two Kaggle datasets: "Dogs vs. Cats" and "Dog Breed Identification."
- Dog images are downloaded from the "Dog Breed Identification" dataset, and cat images are obtained from the "Dogs vs. Cats" dataset.

**Step 2: Label Information**
- Label information is extracted from the "label.csv" file in the "Dog Breed Identification" dataset, containing dog breed labels corresponding to each image.

**Step 3: Dog Breed Separation**
- We analyze the label information to identify unique dog breeds present in the dataset.
- These unique breeds are divided into training and testing sets.

**Step 4: Organize Dog Images by Breed**
- For each breed in the training set, corresponding dog images are moved from the "dog-breed-identification" directory to the "dataset/train/dog" directory.
- For each breed in the testing set, corresponding dog images are moved to the "dataset/test/dog" directory.

**Step 5: Cat Image Separation**
- Cat images are isolated from the "dogs-vs-cats" dataset.
- Cat images are divided into training and testing sets, with 8995 used for training and 3505 for testing.

**Step 6: Organize Cat Images**
- For each image in the training set, corresponding cat images are moved from the "dogs-vs-cats" directory to the "dataset/train/cat" directory.
- For each image in the testing set, corresponding cat images are moved to the "dataset/test/cat" directory.

**Step 7: Final Dataset Structure**
- The resulting dataset structure is organized as follows:
  - dataset
    - train
      - dog
        - ...images
      - cat
        - ...images
    - test
      - dog
        - ...images
      - cat
        - ...images

## Model Training using TensorFlow and PyTorch

### Data Preprocessing

- Cat and dog images are stored in separate directories: "cat_images" and "dog_images."
- Data augmentation is applied to the training dataset to enhance model robustness.

### Model Training (TensorFlow)

**Model 1: EfficientNetV2B2 (Initial Training)**
- EfficientNetV2B2 is loaded without pre-trained weights.
- Global average pooling and a dense layer are added for classification.
- The model is compiled with Adam optimizer and Categorical Crossentropy loss.
- Training is performed on the augmented training data.
- A custom callback, SaveModel, saves the model at the end of each epoch.

**Model 2: EfficientNetV2B2 (Fine-Tuning)**
- EfficientNetV2B2 is loaded with pre-trained ImageNet weights.
- Additional layers are added for classification, and the model is compiled.
- Training includes early stopping, custom callbacks, and checkpoints.

**Model 3: Custom CNN Model**
- A custom CNN model is designed with Conv2D and Dense layers.
- The model is compiled with Adam optimizer and Binary Crossentropy loss.
- Training is performed on the augmented training data, with model-saving callbacks.

**Model 4: Transfer Learning with VGG16**
- VGG16 is loaded without pre-trained weights (excluding top layers).
- Additional layers are added for classification, and the model is compiled.
- Training is performed on the augmented training data, with early stopping and checkpoints.

### Model Training (PyTorch)

**Base Model Initialization**
- EfficientNetV2 model with the "tf_efficientnetv2_b2" architecture is initialized and loaded with both pre-trained and random weights.
- A summary of the model architecture is printed.

**Custom Callback**
- A custom SaveModel callback is defined to save the model after the first epoch, and the saved model is uploaded to Google Cloud Storage.

**Poutyne Callbacks and Training**
- The Poutyne framework is used for training.
- EarlyStopping, ModelCheckpoint, and WandbLogger callbacks are defined.
- An EfficientNetV2 model is created using the poutyne.Model class.
- The model is trained using the fit_generator function, and training history is stored.

## Results

The project evaluates the performance of different models using accuracy as the metric. Here are the results:

| Model                          | Accuracy |
|--------------------------------|----------|
| EfficientNetV2B2 (PyTorch)     | 0.98     |
| EfficientNetV2B2 (PyTorch, No Pretraining) | 0.86     |
| Custom Model (TensorFlow)      | 0.68     |
| EfficientNetV2B2 (TensorFlow)    | 0.59     |
| VGG16 (TensorFlow)             | 0.49     |
| EfficientNetV2B2 (TensorFlow)  | 0.40     |

## Deploying ONNX Models for Cat vs. Dog Prediction using FastAPI

### Model Conversion to ONNX Format

- Pretrained and scratch EfficientNetV2B2 models are loaded using PyTorch.
- Dummy input tensors are defined for ONNX export.
- torch.onnx.export is used to convert both models to ONNX format.
- The ONNX models are saved to specified paths.

### Building the FastAPI Application

- The FastAPI application is built, and necessary libraries are imported.
- Configuration settings for the app are defined.
- ONNX models are loaded using onnxruntime.
- A dictionary maps prediction indices to classes (0: 'Cat', 1: 'Dog').
- Functions to preprocess uploaded images and handle predictions are implemented.
- API endpoints for pretrained and scratch models are created for image uploads and predictions.
- Exception handling is implemented to provide meaningful responses.

### Dockerization for Deployment

- A Dockerfile is created to define the environment for the FastAPI application.
- The official Python image is used as the base image.
- Dependencies are installed, and the FastAPI code is copied into the container.
- The port is exposed, and the FastAPI app is started using CMD.
- A docker-compose.yml file configures the Docker service.

### Using the Deployed Service

- Docker images are built using the Dockerfile and requirements file.
- Docker Compose is used to start the service with the defined configuration.
- The FastAPI app can be accessed through the specified port on the local machine.

## Conclusion

This project takes a fresh approach to the Dogs vs. Cats classification problem by combining datasets, training models using TensorFlow and PyTorch, and deploying models for real-time predictions via FastAPI. The results showcase the performance of various models and demonstrate the potential for combining diverse datasets to create more complex and challenging classification tasks. The deployment of ONNX models with Fast
