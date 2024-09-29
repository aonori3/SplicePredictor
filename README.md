# SplicePredictor

SplicePredictor is a convolutional neural network for genomic sequence classification.

## Setup

1. Clone this repository:
   ```bash
   git clone https://github.com/aonori3/SplicePredictor.git
   cd SplicePredictor
   ```

2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. To train the model:
   ```bash
   python train_splicepredictor.py
   ```

2. To use the trained model for predictions, use the following Python code:
   ```python
   from tensorflow.keras.models import load_model
   
   model = load_model('splicepredictor.h5')
   predictions = model.predict(your_input_data)
   ```

## Model Architecture

The SplicePredictor model uses multiple convolutional blocks followed by dense layers for classification. For more details, see `splicepredictor.py`.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

For any questions or concerns, please contact the author at aoiotani@college.harvard.edu.