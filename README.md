# Sample-Ai-Project
# SAuth2.0 - Handwritten Digit Recognition for Enhanced Security

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

## Overview

SAuth2.0 is an advanced multi-factor authentication system that enhances security by incorporating handwritten digit recognition as a second verification step. This project combines traditional password authentication with machine learning-based handwriting recognition to provide robust security without compromising user privacy.

![MFA Concept](https://piregcompliance.com/wp-content/uploads/2020/03/multifactor-authentication-factors.png)

## Key Features

- **Multi-Factor Authentication**: Combines password (something you know) with handwritten input (something you do)
- **Enhanced Security**: Adds an additional layer of verification beyond passwords
- **Privacy Preservation**: Avoids dependence on biometric data or device-centric methods
- **High Accuracy**: Achieves 96.3% accuracy on training data and 98.48% on external validation data
- **Multiple ML Models**: Implements and compares Random Forest, Logistic Regression, CNN, and LSTM approaches

## Problem Statement

SAuth2.0 addresses existing vulnerabilities in user authentication by establishing an optimal balance between user privacy and sensitive data protection. The system aims to mitigate both current security weaknesses and emerging threats through a multi-layered approach.

## How It Works

1. User enters their password (first authentication factor)
2. System prompts the user to handwrite a specific digit using touch input or stylus
3. Machine learning model processes the handwritten input
4. Authentication is granted or denied based on accurate recognition
5. If recognition fails, the user is prompted to re-enter the digit

## Technical Implementation

### Machine Learning Models

The project implements and compares several machine learning approaches:

| Model | MNIST Accuracy | Kaggle Dataset Accuracy |
|-------|---------------|------------------------|
| Random Forest (Model 1) | 96.3% | 98.46% |
| Random Forest (Model 2) | 95.8% | 98.04% |
| Logistic Regression | 91.2% | 83.96% |
| CNN | 99.76% | 99.56% |
| LSTM | 99.11% | 98.99% |

### Project Structure

```
SAuth2.0/
├── 1_Train_Random_Forest_Model_1.py     # Trains base Random Forest model
├── 2_Train_Random_Forest_Model_2.py     # Trains tuned Random Forest model
├── 3_Train_Logistic_Regression_Model.py # Trains Logistic Regression model
├── 4_Test_All_Models_Kaggle_DataSet.py  # Tests all models on Kaggle dataset
├── 5_Plot_Comparision_Graphs.py         # Generates comparison visualizations
├── 6_Plot_MNIST_Dataset_Details.py      # Analyzes and visualizes MNIST data
├── 7_CNN_LSTM_Model_Train_MNIST.py      # Trains CNN and LSTM models
├── 8_CNN_LSTM_Model_Predict_Kaggle.py   # Tests CNN and LSTM on Kaggle data
├── AI_ML.ipynb                          # Jupyter notebook with all code
├── README.md                            # Project documentation
└── models/                              # Saved model files
    ├── random_forest_mnist.joblib
    ├── random_forest_model_2_mnist.joblib
    ├── logistic_regression_mnist.joblib
    ├── CNN_Model_1.h5
    └── LSTM_Model.h5
```

## Dataset Information

The project utilizes two primary datasets:

1. **MNIST Dataset**: 60,000 training samples and 10,000 testing samples of handwritten digits (28x28 pixels)
2. **Kaggle Dataset**: External validation dataset for comprehensive model testing

## Installation and Usage

### Prerequisites

- Python 3.8+
- TensorFlow 2.x
- scikit-learn
- pandas
- numpy
- matplotlib
- joblib

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/SAuth2.0.git
   cd SAuth2.0
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the Kaggle dataset:
   - Visit https://www.kaggle.com/competitions/digit-recognizer/data
   - Download the `train.csv` file and place it in the project directory

### Running the Code

To train the Random Forest model:
```bash
python 1_Train_Random_Forest_Model_1.py
```

To test all models on the Kaggle dataset:
```bash
python 4_Test_All_Models_Kaggle_DataSet.py
```

To visualize dataset details:
```bash
python 6_Plot_MNIST_Dataset_Details.py
```

To train CNN and LSTM models:
```bash
python 7_CNN_LSTM_Model_Train_MNIST.py
```

## Results and Findings

After rigorous testing and evaluation, the project demonstrated:

1. CNN model achieved the highest accuracy (99.56% on Kaggle dataset)
2. Random Forest models performed exceptionally well (98.46% accuracy) with lower computational requirements
3. Logistic Regression provided a solid baseline (83.96% accuracy) but proved less effective for this application
4. All models except Logistic Regression maintained or improved performance when tested on the external Kaggle dataset

![Model Comparison](https://i.imgur.com/placeholder.png)

## Future Work

Building on the success of SAuth2.0, future development will focus on:

1. **SAuth4.0**: Enhancing security by recognizing user-specific handwriting styles
2. **Multi-language Support**: Extending capabilities to other scripts like Devanagari for Hindi
3. **Improved Accessibility**: Adapting to alternative input methods for inclusive user experience

## Lessons Learned

Key takeaways from this project include:

1. The importance of careful parameter selection for model accuracy
2. The value of testing models on multiple datasets
3. The balance between model complexity and performance
4. The impact of data quality on recognition accuracy

## References

1. Pirege Compliance. (2020). The Importance of Multifactor Authentication for Compliance and Safety. [Link](https://piregcompliance.com/authentication/the-importance-of-multifactor-authentication-for-compliance-and-safety/)
2. TensorFlow. (2023). Keras: The high-level API for TensorFlow. [Link](https://www.tensorflow.org/guide/keras)
3. Kaggle. (2021). Dataset for SAuth4.0. [Link](https://www.kaggle.com/competitions/digit-recognizer/data)
4. Kaggle. Dataset for Hindi Character Recognition. [Link](https://www.kaggle.com/datasets/suvooo/hindi-character-recognition)
5. Medium. (2019). Dataset and information on Devanagari. [Link](https://medium.com/analytics-vidhya/cpar-hindi-digit-and-character-dataset-1347a7ff946)

## Author

**Shivansh Mishra**  


## License

This project is licensed under the MIT License - see the LICENSE file for details.
