# Token Reducer

> **Note:** This project is currently under active development. Features and usage may change.

Token Reducer is a Python CLI tool that automatically removes polite expressions and greetings from text. It is especially useful for preprocessing in NLP applications or reducing token count.

## Features

- Automatically removes polite expressions (e.g., "thanks", "best regards", "please", etc.)
- Cleans up greetings and sign-offs
- Easy to use from the command line
- Can be integrated into pipelines with stdin/stdout
- Model is trained on the Intel/polite-guard dataset

## Technologies & Algorithms

- **scikit-learn**: Used for feature extraction and model training
- **joblib**: For model serialization
- **datasets (Hugging Face)**: For loading the Intel/polite-guard dataset
- **Regular Expressions (re module)**: For pattern matching and text cleaning
- **CountVectorizer**: For extracting n-gram features from text
- **Log-odds Scoring**: To identify and rank polite expressions

## Installation

1. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
2. The model file (politeness_model.joblib) is provided. If you want to train your own model:
   ```bash
   python train_model.py
   ```

## Usage

The strip function is in strip_polite.py. Example usage:

```python
from strip_polite import strip_polite

text = "Hello World, congratulations on taking the first step towards improving!"
result = strip_polite(text)
print(result)  # Output: 'Hello World'
```
## Files
- `strip_polite.py`: Main function to remove polite expressions
- `train_model.py`: Trains the model and generates the regex pattern
- `politeness_model.joblib`: Trained model and regex pattern
- `requirements.txt`: Required Python packages

## License
MIT License
