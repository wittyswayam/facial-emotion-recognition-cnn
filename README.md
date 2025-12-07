## 😠 Human Facial Emotion Recognition using CNN

This repository features a complete deep learning project for **Facial Emotion Recognition**, which classifies human emotions from images. The system employs a custom-designed **Convolutional Neural Network (CNN)** trained to distinguish between **seven core emotional states**.

The project is structured with a training pipeline (Jupyter Notebook) and a functional, deployable web application built using the **Flask** framework.

-----

### 🧠 Detailed Deep Learning Architecture

The model is a custom, sequential CNN explicitly designed for the task of image-based emotion classification.

#### Custom CNN Layers

The architecture is constructed from a stacked sequence of layers tailored for image feature extraction and processing:

  * **Convolutional (Conv2D) Layers:** These layers act as the primary feature extractors. They apply a set of learned filters across the input image to detect hierarchical features, starting from low-level patterns like **edges and corners** in early layers, to high-level features like **facial contours and landmark regions** in deeper layers.
  * **Pooling (MaxPooling2D) Layers:** These layers follow the convolutional layers and perform **downsampling**. By taking the maximum value within a filter region, they reduce the spatial size of the feature maps, which helps reduce computational load and makes the model more robust to minor variations or shifts in the input image (translational invariance).

[Image of CNN layers diagram]

#### Regularization and Optimization

To ensure training stability and prevent the model from simply memorizing the training examples (overfitting), key regularization and optimization techniques are used:

  * **Batch Normalization:** Applied after convolutional layers to normalize the input for the next layer. This significantly **stabilizes and accelerates training** by maintaining consistent mean and variance across mini-batches.
  * **Dropout:** A regularization method where a percentage of neurons are randomly ignored during each training step. This forces the remaining neurons to learn more **robust and distributed representations**, improving the model's ability to generalize to new, unseen faces.
  * **Advanced Callbacks:** During training, **Early Stopping** (to stop when validation performance plateaus) and **ReduceLROnPlateau** (to dynamically lower the learning rate) are used to manage the learning process and prevent the model from prematurely settling in a suboptimal state.

#### Classification Output

The network's final stage converts the extracted features into a classification result:

  * A **Flatten** layer converts the 2D feature maps into a single 1D vector.
  * The final layer is a **Dense layer with 7 units**, corresponding to the number of classes. It uses the **Softmax activation function**, which outputs a probability distribution where the sum of probabilities for the seven emotion classes (`angry`, `disgust`, `fear`, `happy`, `neutral`, `sad`, `surprise`) equals one. The highest probability determines the predicted emotion.

-----

### 📊 Data Pipeline and Robustness

#### Training Data Standardization

The model was trained on the **face-expression-recognition-dataset** (similar to FER-2013) from Kaggle, comprising over 28,000 training images. Critical standardization steps were performed:

  * Images were resized to a small, computationally efficient size of **(48, 48) pixels**.
  * Images were converted to **grayscale** (single channel) to focus the model on key facial features and reduce data dimensionality, as color information is less relevant for emotion classification.

#### Data Augmentation

To boost the model's ability to generalize beyond the exact training examples, extensive **Data Augmentation** was applied using Keras's `ImageDataGenerator`:

  * Random transformations like **rotation, zooming, and horizontal flipping** were applied to the images. This artificially increases the effective size and diversity of the dataset, making the model more robust to variations in real-world camera angles and facial positioning.

-----

### 🚀 Setup and Deployment

#### Running the Project

To set up and run the web application locally, follow these steps:

```bash
# Clone the repository
git clone https://github.com/wittyswayam/facial-emotion-recognition-cnn.git
cd facial-emotion-recognition-cnn/facial emotion recognition cnn

# Install the necessary dependencies
# ( tensorflow, keras, flask, numpy, etc. are listed in a requirements.txt file)
pip install -r requirements.txt

# Run the Flask web application
python main.py
```

The application will launch on a local server (typically `http://127.0.0.1:5000`), where you can upload an image to receive an emotion prediction.

#### Flask Web Deployment

The system is made interactive via a lightweight web server using **Flask**. The `main.py` file loads the trained model and defines the application logic:

  * It handles the **POST request** of an image file upload.
  * The image is converted to the required **(48, 48) grayscale** format.
  * The model runs a prediction and returns the **predicted emotion class** and its **confidence score** to the user via the `index.html` interface.

-----

### 💡 Future Works

To elevate the project to an industry-standard application, focus should be placed on robust image processing and leveraging state-of-the-art models.

#### Integrate a Face Detector and Pre-Processing Pipeline

The current model relies on the input image being a pre-cropped, centered face. This is highly impractical for real-world use (e.g., webcam or social media photos). The system should be enhanced with a **mandatory two-stage pipeline**:

  * **Stage 1: Face Detection:** Integrate a dedicated, fast detection algorithm (like **MTCNN** or **Dlib's face detector**) to locate the bounding box of a face within any arbitrary image.
  * **Stage 2: Emotion Classification:** Pass the **cropped face region** to the emotion CNN. This ensures the classifier only analyzes the relevant pixels, making the system far more **reliable** and robust to complex images with multiple subjects or distractions.
