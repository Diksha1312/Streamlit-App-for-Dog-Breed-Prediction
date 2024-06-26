# Streamlit-App-for-Dog-Breed-Prediction

# Dog Breed Identification using Convolutional Neural Network (CNN)

This project implements a Convolutional Neural Network (CNN) using Keras and TensorFlow to identify the breed of a dog from an input image. This is a supervised machine learning task, specifically a multiclass classification problem.

## Setup

### Create Conda Environment

1. **Clone the Repository**

   ```bash
   git clone https://github.com/Diksha1312/Streamlit-App-for-Dog-Breed-Prediction
   cd Streamlit-App-for-Dog-Breed-Prediction

2. **Create Conda Environment**

    ```bash
   conda create --n env python=3.8
   conda activate env

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt


## Steps Implemented

### Data Acquisition from Kaggle
- The dataset containing dog images and corresponding labels (breeds) was obtained from Kaggle.

### Label Preparation
- A CSV file containing image IDs and their respective breed labels was loaded to associate each image with its breed.

### Data Exploration
- The distribution of dog breeds in the dataset was analyzed to understand the dataset's class balance.

### One-Hot Encoding (OHE)
- The breed labels were one-hot encoded to transform categorical labels into a format suitable for model training.

### Image Loading and Preprocessing
- Dog images were loaded, converted into arrays, and normalized to ensure consistent input to the model.

### Data Validation
- Checks were performed on the dimensions and size of the input data arrays (X) and their corresponding labels (Y).

### Model Architecture Design
- A CNN architecture was designed using Keras to learn features from the input images and predict the breed labels.

### Model Training
- The dataset was split into training and validation sets to train the model. An accuracy plot was generated to visualize model performance during training.

### Model Evaluation
- The trained model was evaluated on the validation set to determine its accuracy in predicting dog breeds.

### Prediction Using the Model
- The trained model was utilized to make predictions on new dog images, enabling identification of the breed from unseen images.

## Integration with Streamlit App

This project also includes a Streamlit web application for interactive dog breed prediction. Users can upload images of dogs, and the app will display the uploaded image along with the predicted breed using the trained CNN model.

### How to Run the Streamlit App
1. **Install Dependencies**
   - Ensure Python, Streamlit, TensorFlow, and Keras are installed in your environment.

2. **Run the Application**
   - Navigate to the directory containing `main.py`.
   - Activate Conda Environment
     ```bash
     conda activate env
   - Execute `streamlit run main.py` in your terminal or command prompt.
   - Access the app through the provided local URL in your web browser.

## Potential Applications

- **Animal Welfare Organizations**
  - NGOs and shelters can use this model to identify dog breeds accurately, aiding in their rescue and adoption efforts.

- **Educational Purposes**
  - Educational institutions can utilize the app to teach students about image classification and machine learning concepts.

