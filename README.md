🔢 MNIST Handwritten Digit Recognition
Show Image
📝 Project Description
This project implements a deep learning model using Convolutional Neural Networks (CNN) to recognize handwritten digits from the MNIST dataset. The model classifies grayscale images of digits (0-9) with high accuracy.
✨ Features

🧠 Uses CNN architecture for image classification
📊 Trains on the MNIST dataset
📈 Evaluates model performance with accuracy and loss metrics
🖼️ Supports prediction on custom handwritten digit images

📋 Table of Contents

Installation
Usage
Dataset
Model Architecture
Results
Contributing
License

🔧 Installation

Clone the repository
bashgit clone https://github.com/yourusername/mnist-digit-recognition.git
cd mnist-digit-recognition

Install required dependencies:
bashpip install -r requirements.txt


🚀 Usage
🏋️ Training the Model
Run the training script:
bashpython train.py
📊 Evaluating the Model
Evaluate the model on the test set:
bashpython evaluate.py
🔍 Making Predictions
Predict digits on custom images:
bashpython predict.py --image path/to/your/image.png
📚 Dataset
The MNIST dataset is automatically downloaded via TensorFlow/Keras datasets module. It contains:

6️⃣0️⃣,0️⃣0️⃣0️⃣ training images
1️⃣0️⃣,0️⃣0️⃣0️⃣ test images
2️⃣8️⃣x2️⃣8️⃣ pixel grayscale images
1️⃣0️⃣ classes (digits 0-9)

🏗️ Model Architecture
The CNN model consists of:

🔍 Convolutional layers for feature extraction
🏊 Max pooling layers for dimensionality reduction
🎭 Dropout layers to prevent overfitting
🧠 Dense layers for classification

📊 Results
The model achieves around 9️⃣9️⃣% accuracy on the test set.
Example confusion matrix:
[[ 980    0    1    0    0    1    1    0    1    0]
 [   0 1134    1    0    0    0    0    0    0    0]
 [   2    2 1026    0    0    0    0    2    0    0]
 [   0    0    0 1009    0    3    0    1    3    0]
 [   0    0    0    0  978    0    1    0    0    3]
 [   1    0    0    3    0  886    2    0    0    0]
 [   3    1    0    0    1    4  949    0    0    0]
 [   0    1    5    0    0    0    0 1023    0    3]
 [   0    0    0    1    1    1    0    0  972    0]
 [   0    0    0    0    2    1    0    3    0  994]]
