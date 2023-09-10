# Dataset Preprocessing

## Initial Dataset

The initial dataset used for this project contained images with 7 emotion classes:

- Happy
- Angry
- Neutral
- Surprised
- Fear
- Sad
- Disgust

In total, there were nearly 39,000 images in the initial dataset.
original  dataset  link:

## Pre-processing Steps

To prepare the dataset for use in a music recommendation system, several pre-processing steps were performed:

### 1. Reducing Complexity

To simplify the dataset and reduce its complexity, three classes were removed:

- Surprised
- Fear
- Disgust

After removing these classes, the dataset was left with a total of 26,000 images.

### 2. Grayscale Conversion

All the images in the dataset were converted to grayscale. This conversion reduced the dimensionality of the images and made them more suitable for further processing.

### 3. Resizing Images

The images were scaled down to a uniform size of 48x48 pixels. This resizing step ensured that all images had the same dimensions, making it easier to feed them into a machine learning model.

### 4. Normalization

The pixel values of the grayscale images were normalized to a range between 0 and 1. This step is crucial for training machine learning models as it helps in improving convergence during training and ensures that all features have similar scales.

### 5. Label Encoding

The emotion classes in the dataset were label encoded using the Scikit-Learn library. Label encoding assigns a unique numerical label to each class, which is necessary for training machine learning models.

These pre-processing steps resulted in a cleaned and prepared dataset that could be used for building a music recommendation system based on emotions.

The processed dataset can be found in this repository under the appropriate directory or file.

Please note that the exact code and details of the preprocessing may vary depending on the programming language and libraries used for the project.

