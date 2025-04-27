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
> Please help me with this task, thank you
Cleaned Sentence:
help me with this task

> exit
Goodbye!
```

The CLI uses Rich for colored output to enhance readability of the results.

## Files
- `strip_polite.py`: Main function to remove polite expressions
- `train_model.py`: Trains the model and generates the regex pattern
- `politeness_model.joblib`: Trained model and regex pattern
- `requirements.txt`: Required Python packages

## License
MIT License
