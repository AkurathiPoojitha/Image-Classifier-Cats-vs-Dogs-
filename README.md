# Image-Classifier-Cats-vs-Dogs-
**Project Overview**
The goal of this project is to train a CNN model on labeled images of cats and dogs and use the trained model to predict the class of unseen images. The model automatically learns visual features such as edges, shapes, and textures from images.

**🛠️ Technologies Used**

Python 3.13

PyTorch

Torchvision

Pillow

VS Code

**📂 Project Structure**
cats_vs_dogs/
│
├── dataset/
│   └── train/
│       ├── cats/
│       └── dogs/
│
├── train.py
├── predict.py
├── cat_dog_model.pth
└── test.jpg

**📊 Dataset**

The dataset consists of labeled images of cats and dogs downloaded from Kaggle.
Images are organized into separate folders (cats and dogs) which are automatically used as class labels during training.

**⚙️ How It Works**
1️⃣ Training the Model

Images are resized and converted into tensors

A CNN model is trained using Cross Entropy Loss and Adam optimizer

The trained model is saved as cat_dog_model.pth

**Run:**

python train.py

2️⃣ Predicting an Image

The saved model is loaded

A test image (test.jpg) is passed to the model

The model predicts whether the image is a Cat or a Dog

Run:

python predict.py

✅ Output Example
Prediction: Cat 🐱


or

Prediction: Dog 🐶

**📈 Results**

The CNN model successfully learned to differentiate between cat and dog images. Training loss decreased across epochs, and the model was able to correctly classify unseen images, demonstrating effective feature learning.

**🚀 Future Improvements**

Increase dataset size for better accuracy

Apply data augmentation techniques

Use advanced CNN architectures like ResNet or MobileNet

Deploy the model as a web or mobile application

**📚 What I Learned**

Basics of Convolutional Neural Networks

Image preprocessing techniques

Training and saving deep learning models in PyTorch

Using trained models for prediction
