# AI-Driven Classification of Lung Cancer Stages in CT Imaging

## Project Overview
This project leverages deep learning techniques to classify lung cancer stages from CT scan images. Using the **IQ-OTH/NCCD Lung Cancer Dataset**, the model distinguishes between **normal, benign, and malignant** cases. The goal of this project is to aid in early diagnosis and improve clinical decision-making in lung cancer treatment.

## Dataset
The dataset, provided by the **Iraq-Oncology Teaching Hospital/National Center for Cancer Diseases (IQ-OTH/NCCD)**, contains:
- **Total Images**: 1190 CT scan slices
- **Cases**: 110 cases, classified into three categories:
  - **Normal**: 55 cases
  - **Benign**: 15 cases
  - **Malignant**: 40 cases

Each scan comprises multiple slices (80 to 200), representing the human chest at various angles. The images were originally collected in **DICOM** format and converted for analysis.

### Dataset Details
- **Source**: SOMATOM Siemens CT scanner
- **CT Protocol**: 120 kV, slice thickness of 1 mm
- **Window Width**: 350 to 1200 HU
- **Window Center**: 50 to 600
- **Inspiration**: Full breath-hold at inspiration

## Model Architecture
This project utilizes a Convolutional Neural Network (CNN) model implemented in TensorFlow and Keras to classify the CT images. The model architecture is as follows:
1. **Convolutional Layers** with ReLU activation
2. **Max Pooling Layers** for down-sampling
3. **Flatten Layer** to convert the 2D matrices to 1D vectors
4. **Dense Layers** with Dropout for regularization
5. **Output Layer** with softmax activation for multi-class classification

### Model Summary
- **Input Shape**: (128, 128, 1) for grayscale images
- **Output Classes**: Normal, Benign, Malignant

## Important Note
The trained model file, `cancer_classification_model.keras`, is not included in this repository because it exceeds GitHub's file size limit (25 MB). To obtain this model, simply run the `lung_cancer.ipynb` notebook. This will train the model on the dataset and save it as `cancer_classification_model.keras` in the `models` directory.

## Installation

To run this project, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/lung-cancer-classification.git
   cd lung-cancer-classification
   

**Install the required packages**:
```bash
    pip install -r requirements.txt
```
**Streamlit App**
```bash
    streamlit run app.py
```
