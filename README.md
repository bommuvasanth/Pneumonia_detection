# 🩺 **Pneumonia Detection using Deep Learning**

## 📘 **Overview**
This project is a deep learning-based medical image analysis system designed to detect **Pneumonia** from **chest X-ray images**.  
It leverages **Convolutional Neural Networks (CNN)** for image classification and integrates **Grad-CAM visualization** to highlight infected regions — enhancing interpretability and trust in predictions.

## 🚀 **Features**
- ✅ **Detects Normal and Pneumonia cases** from X-ray images  
- 🔍 **Integrated Grad-CAM** to visualize infection regions  
- 🧩 **Image preprocessing pipeline** using OpenCV and NumPy  
- ⚠️ **Handles false negatives** using threshold-based logic and fallback prediction  
- 💾 **Stores prediction results** in the database for future reference  
- 🖥️ **Streamlit dashboard** for user-friendly visualization  
- ⚙️ **Optimized model performance** with data augmentation and dropout regularization  

## 🧠 **Tech Stack**
- **Programming Language:** Python  
- **Libraries:** TensorFlow / Keras, OpenCV, NumPy, Matplotlib  
- **Explainability:** Grad-CAM  
- **Database:** MongoDB (stores predictions and analytics)  
- **API:** FastAPI for model access  
- **Dashboard:** Streamlit  
- **Model Type:** Convolutional Neural Network (CNN)  

## ⚙️ **Workflow**
1. **Dataset Loading** → Import chest X-ray dataset and split into train/test sets  
2. **Preprocessing** → Resize, normalize, and augment images for better generalization  
3. **Model Training** → Train CNN model with dropout and ReLU activation layers  
4. **Evaluation** → Measure accuracy, precision, recall, and F1 score  
5. **Explainability** → Generate Grad-CAM heatmaps for each prediction  
6. **False Negative Fix** → Implement logic to minimize misclassification of Pneumonia as Normal  
7. **Database Storage** → Store all predictions and analytics in MongoDB  
8. **Visualization** → Display results on the Streamlit dashboard  

## 📊 **Results**
- 📈 **High accuracy** in detecting Pneumonia vs. Normal cases  
- 🧠 **Grad-CAM visualizations** clearly highlighted infected lung regions  
- 🩹 **Reduced false negatives** through preprocessing enhancements and threshold tuning  
- 💾 **All results stored** in MongoDB for future analysis  
- 🖥️ **Streamlit dashboard** provides easy access to predictions and visualizations  

## 🌟 **Future Enhancements**
- 🚑 **Deploy the model** for real-time X-ray prediction in hospitals  
