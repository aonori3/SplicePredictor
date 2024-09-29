# SplicePredictor

SplicePredictor is a convolutional neural network for genomic sequence classification.

## Setup

1. Clone this repository:
   ```
   git clone https://github.com/aonori3/SplicePredictor.git
   ```

2. Install required packages:
   ```
   pip install tensorflow numpy pandas scikit-learn
   ```

## Usage

1. To train the model:
   ```
   python train_splicepredictor.py
   ```

2. To use the trained model for predictions, load it in your Python script:
   ```python
   from tensorflow.keras.models import load_model
   
   model = load_model('splicepredictor.h5')
   predictions = model.predict(your_input_data)
   ```

## Model Architecture

The SplicePredictor model uses multiple convolutional blocks followed by dense layers for classification. For more details, see `splicepredictor.py`.
