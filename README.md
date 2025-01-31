# Federated-learning-and-Explainable-Ai-based-Tumor-Prediction

This project implements a **hybrid deep learning framework** that integrates **ResNet50** and **Fast Fourier Transform (FFT)** for enhanced feature extraction in brain tumor detection. The baseline **CNN model** achieved **86.46% accuracy**, while **ResNet50** alone improved it to **90.24%**. **FFT combined with **Resnet** gave better results. The **proposed federated learning architecture** further optimized performance, achieving a **final accuracy of 96.07%** by leveraging decentralized training across multiple healthcare institutions.  

To enhance interpretability, **Explainable AI (XAI) techniques, LIME and SHAP**, were integrated to highlight model decision-making areas, ensuring alignment with clinically relevant tumor features. The federated approach not only enhances generalization but also **preserves patient data privacy** by training models locally without sharing raw medical images, addressing compliance with **HIPAA and GDPR** standards.  

## **Technologies Used**  
- `tensorflow_federated` – Federated learning implementation  
- `tensorflow` – Deep learning framework for training models  
- `keras` – High-level API for building CNN and ResNet50 models  
- `torch` – For handling tensors and neural network computations (if PyTorch is used)  
- `torchvision` – Pre-trained model support (ResNet50)  
- `opencv (cv2)` – Image preprocessing (grayscale conversion, contour detection, cropping)  
- `shap` – SHAP (Shapley Additive Explanations) for feature importance visualization  
- `lime` – LIME (Local Interpretable Model-Agnostic Explanations) for explainability in image classification  
