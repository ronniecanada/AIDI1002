
# Brain Tumor Detection Using CNN

This project focuses on detecting brain tumors using deep learning techniques. It uses a convolutional neural network (CNN) to classify MRI images into categories (e.g., normal or abnormal). Transfer learning with VGG16 is also explored for enhanced accuracy.

## **Dataset**
The dataset contains MRI scans categorized into various tumor types, including normal and abnormal classes. The dataset is extracted from the provided ZIP file.

## **Project Structure**
```
|-- brain_tumor_detection
    |-- data/
        |-- archive.zip
    |-- notebooks/
        |-- brain_tumor_detection.ipynb
    |-- src/
        |-- load_data.py
        |-- train_model.py
        |-- predict.py
    |-- outputs/
        |-- brain_tumor_detection_model.h5
    |-- README.md
```

- `data/`: Contains the dataset ZIP file.
- `notebooks/`: Contains Jupyter notebook for full implementation.
- `src/`: Contains Python scripts for data loading, model training, and predictions.
- `outputs/`: Stores the trained model file.
- `README.md`: This file.

## **Setup Instructions**
### **Step 1: Clone the Repository**
```bash
git clone https://github.com/your-repo/brain_tumor_detection.git
cd brain_tumor_detection
```

### **Step 2: Install Dependencies**
Install the required libraries:
```bash
pip install tensorflow keras numpy matplotlib scikit-learn
```

### **Step 3: Extract Dataset**
Ensure the dataset ZIP file is in the `data/` directory. Extract it as follows:
```bash
unzip data/archive.zip -d data/
```

### **Step 4: Run the Code**
Run the Jupyter notebook:
```bash
jupyter notebook notebooks/brain_tumor_detection.ipynb
```

### **Step 5: Train the Model**
The training code is in `train_model.py`. Run it:
```bash
python src/train_model.py
```

### **Step 6: Make Predictions**
Use `predict.py` to classify new images:
```bash
python src/predict.py --image_path /path/to/image.jpg
```

## **Results**
- Test Accuracy: ~98% (CNN), ~100% (VGG16 Transfer Learning)
- Evaluation metrics include accuracy, precision, recall, and F1-score.

## **Future Improvements**
- Incorporate more diverse datasets to improve generalizability.
- Experiment with additional deep learning architectures (e.g., ResNet, EfficientNet).
- Deploy the model for real-time prediction using Flask or FastAPI.
