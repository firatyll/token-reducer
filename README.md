# Token Reducer

```
 ____        ___        __                                           
/\  _`\     /\_ \    __/\ \__                                        
\ \ \L\ \___\//\ \  /\_\ \ ,_\    __    ___      __    ____    ____  
 \ \ ,__/ __`\\ \ \ \/\ \ \ \/  /'__`\/' _ `\  /'__`\ /',__\  /',__\ 
  \ \ \/\ \L\ \\_\ \_\ \ \ \ \_/\  __//\ \/\ \/\  __//\__, `\/\__, `\
   \ \_\ \____//\____\\ \_\ \__\ \____\ \_\ \_\ \____\/\____/\/\____/
    \/_/\/___/ \/____/ \/_/\/__/\/____/\/_/\/_/\/____/\/___/  \/___/ 
                                                                     
                                                                     
 ____    __                                               
/\  _`\ /\ \__         __                                 
\ \,\L\_\ \ ,_\  _ __ /\_\  _____   _____      __   _ __  
 \/_\__ \\ \ \/ /\`'__\/\ \/\ '__`\/\ '__`\  /'__`\/\`'__\
   /\ \L\ \ \ \_\ \ \/ \ \ \ \ \L\ \ \ \L\ \/\  __/\ \ \/ 
   \ `\____\ \__\\ \_\  \ \_\ \ ,__/\ \ ,__/\ \____\\ \_\ 
    \/_____/\/__/ \/_/   \/_/\ \ \/  \ \ \/  \/____/ \/_/ 
                              \ \_\   \ \_\               
                               \/_/    \/_/               
```

> **Note:** This project is currently under active development. Features and usage may change.

Token Reducer is a Python CLI tool that automatically removes polite expressions and greetings from text. It is especially useful for preprocessing in NLP applications or reducing token count.

## Features

- Automatically removes polite expressions (e.g., "thanks", "best regards", "please", etc.)
- Cleans up greetings and sign-offs
- Easy to use from the command line
- Can be integrated into pipelines with stdin/stdout
- Model is trained on the Intel/polite-guard dataset
- Provides detailed reports of removed polite expressions

## Technologies & Algorithms

- **scikit-learn**: Used for feature extraction and model training
- **datasets (Hugging Face)**: For loading the Intel/polite-guard dataset
- **CountVectorizer**: For extracting n-gram features from text
- **Log-odds Scoring**: To identify and rank polite expressions
- **spaCy**: For text lemmatization and linguistic preprocessing
- **pickle**: For model serialization

We have implemented multiple approaches to politeness detection and removal:

1. **Original Approach**: Used `CountVectorizer` and log-odds scoring to select the top 250 polite n-grams and build a regex-based stripper.
2. **TF–IDF + Logistic Regression pipeline**: Added a `TfidfVectorizer` followed by an `LogisticRegression` model inside a Scikit‑learn `Pipeline`, enabling:
   * Weighted n-gram features (TF–IDF) for more nuanced text representation.
   * L2 regularization in logistic regression to prevent overfitting.
   * Single-step `fit`/`predict` calls and seamless integration with `GridSearchCV` for hyperparameter tuning.
3. **Direct Feature Matching Approach**: Our latest implementation uses:
   * Stored dictionary of polite features with their importance scores
   * SpaCy-powered lemmatization to match word variants
   * Advanced n-gram detection for multi-word polite expressions
   * Flexible partial matching to catch related expressions
   * Returns both cleaned text and a list of removed expressions
   * Politeness score calculation based on number of detected polite expressions

The direct feature matching approach offers several advantages:
- More transparent results (shows exactly what was removed)
- Better handling of context through n-gram detection
- More flexible matching through lemmatization
- Simpler configuration through adjustable thresholds

## Performance Metrics

Below are the evaluation results comparing the **TF–IDF + LR pipeline** against the **log-odds classifier** on a held-out test set:

### Confusion Matrix: Pipeline

![Confusion Matrix: Pipeline](<assets/confusion matrix pipeline.png>)

### Confusion Matrix: Log-Odds

![Confusion Matrix: Log-Odds](<assets/confusion matrix log odds.png>)

### Precision-Recall Curves

![Precision-Recall Curves](<assets/precision recall curves.png>)

### ROC Curves

![ROC Curves](<assets/ROC curves.png>)

| Metric                | Pipeline (TF–IDF + LR) | Log-Odds Classifier | Direct Feature Matching |
| --------------------- | ---------------------- | ------------------- | ----------------------- |
| **Accuracy**          | 95%                    | 93%                 | 94%                     |
| **Macro F1-Score**    | 0.95                   | 0.93                | 0.94                    |
| **ROC AUC**           | 0.989                  | 0.973               | N/A                     |
| **Average Precision** | 0.99                   | 0.97                | N/A                     |
| **Interpretability**  | Low                    | Medium              | High                    |

The direct feature matching approach provides excellent balance between accuracy and interpretability of results.

## Installation

1. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
2. Install spaCy model:
   ```bash
   python -m spacy download en_core_web_sm
   ```
3. Ensure model directory exists:
   ```bash
   mkdir -p model
   ```
4. Train the model (if not using pre-trained model):
   ```bash
   python train_model.py
   ```

## Usage

### Command Line Interface (CLI)

The project includes a user-friendly command-line interface that allows you to interactively strip polite expressions from text:

1. Run the CLI tool:
   ```bash
   python CLI.py
   ```

2. The CLI will display a welcome banner and prompt you to enter sentences.

3. Type your text with polite expressions and press Enter to see the cleaned version.

4. Type 'exit' when you want to quit the program.

Example session:
```
>  Could you please help me with this task, thank you

Politeness Score: 0.2000

Cleaned Sentence:
Could you help me with this task

Removed Features:
- please
- thank you

> exit
Goodbye!
```

The CLI uses Rich for colored output to enhance readability of the results.

## Files
* `strip_polite.py`: Main function to remove polite expressions and extract features
* `train_model.py`: Trains the model and generates the polite features dictionary
* `model/polite_features.pkl`: Trained model with polite expression weights
* `CLI.py`: Interactive command-line interface
* `evaluate.py`: Script for detailed performance evaluation and visualization
* `requirements.txt`: Required Python packages

## License
This project is licensed under the [MIT License](LICENSE).
