# 🐶🐱 Dog vs Cat Classification using TensorFlow and MobileNet
This project implements a neural network model that classifies images as either dogs or cats. The model uses the MobileNetV2 architecture, which is a pre-trained model available via TensorFlow Hub. The dataset for training and testing is extracted using the Kaggle API and consists of the famous "Dogs vs Cats" dataset.

## 🛠️ Prerequisites
- Python 3.x
- Kaggle API credentials (Kaggle JSON file)
- Libraries:
- numpy
- pandas
- tensorflow==2.14
- tensorflow-hub==0.15
- Pillow
- opencv-python
- matplotlib

  
Install the required dependencies:

```
pip install numpy pandas tensorflow==2.14 tensorflow-hub==0.15 Pillow opencv-python matplotlib
```

## 📂 Dataset
We use the Dogs vs Cats dataset from Kaggle. You will need to download it using the Kaggle API.

### Steps to Download Dataset Using Kaggle API:
Install the Kaggle library:
```
pip install kaggle
```
Configure the Kaggle API credentials:

Place your kaggle.json file in the right location:
```
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```
Download the dataset:

```
mpetitions download -c dogs-vs-cats
```
Extract the dataset using Python's ZipFile module:

```
from zipfile import ZipFile
dataset = '/content/dogs-vs-cats.zip'
with ZipFile(dataset, 'r') as zip:
    zip.extractall()
    print('Dataset extracted successfully!')
```
Extract the train.zip file as well:


```
with ZipFile('/content/train.zip', 'r') as zip:
    zip.extractall()
    print('Training images extracted successfully!')
```
## 📈 Project Workflow
1. Image Preprocessing
Resizing: Resize all images to 224x224 pixels to match the input size required by MobileNet.
Scaling: Normalize the pixel values to a range of 0-1 by dividing by 255.
2. Labeling Images
Assign 0 for cat images and 1 for dog images.
Store these labels in a list for later use.
3. Splitting the Dataset
Split the dataset into training and testing sets using an 80-20 ratio with the train_test_split function from Scikit-learn.
4. Model Architecture
The model uses MobileNetV2 as a feature extractor, which is pre-trained on ImageNet.
A final dense layer with 2 output units (for cats and dogs) is added to predict the class of the image.
5. Model Training and Evaluation
The model is trained for 5 epochs using the Adam optimizer and Sparse Categorical Crossentropy loss function.
Evaluate the model on the test set to calculate the accuracy.
6. Predictive System
You can input an image of a dog or cat, and the model will classify the image and output whether it represents a cat or a dog.

## Model
Architecture: MobileNetV2 pre-trained on ImageNet.
Framework: TensorFlow and TensorFlow Hub.

## 🔗 Useful Links
[Kaggle API Documentation](https://www.kaggle.com/docs/api)
[TensorFlow Hub: MobileNetV2](https://www.kaggle.com/models/google/mobilenet-v2/tensorFlow2/tf2-preview-feature-vector/4?tfhub-redirect=true)
[Dogs vs Cats Dataset](https://www.kaggle.com/c/dogs-vs-cats)
