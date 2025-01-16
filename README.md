# Real-Estate-Price-Prediction-using-Deep-Learning

## Overview

This project utilizes deep learning techniques to predict real estate prices based on various property features. By training a neural network model on historical data, the system aims to provide accurate price estimations for real estate properties.

---

## Features

- **Data Preprocessing**: Cleans and prepares the dataset for training.
- **Model Training**: Implements a deep neural network using TensorFlow and Keras.
- **Evaluation**: Assesses model performance using metrics like Mean Absolute Error (MAE).
- **Prediction**: Generates price predictions for new property data.

---

## Installation

### 1. Clone the Repository

Run the following command in your terminal to clone the project:
```bash
git clone https://github.com/sneha30404/Real-Estate-Price-Prediction-using-Deep-Learning.git
cd Real-Estate-Price-Prediction-using-Deep-Learning
```

### 2. Install Dependencies

Make sure you have Python 3.8 or later installed. Then, install the required packages:
```bash
pip install -r requirements.txt
```

---

## Usage

### 1. Prepare the Dataset
- Place your dataset file (`realestate_prices.csv`) in the project directory.

### 2. Run the Training Script
To preprocess the data, train the model, and save the trained model to disk:
```bash
python train_model.py
```

### 3. Make Predictions
Use the trained model to make predictions on new data:
```bash
python predict.py --input new_data.csv --output predictions.csv
```

---

## Contributing

Contributions are welcome! Please fork the repository and create a new branch for any feature additions or bug fixes. Submit a pull request for review.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## Contact

For any questions or support, please contact:
- **Name**: Sneha
- **GitHub**: [sneha30404](https://github.com/sneha30404)

