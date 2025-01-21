import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional, GlobalMaxPooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import pickle
import os
from tkinter import Tk, filedialog
from tensorflow.keras.mixed_precision import set_global_policy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.mixed_precision import LossScaleOptimizer

# Enable mixed precision globally
set_global_policy('mixed_float16')
tf.config.experimental.enable_tensor_float_32_execution(False)

# Global Variables
MAX_WORDS = 10000  # Maximum number of words to keep based on frequency
MAX_LEN = 100  # Maximum length of sequences (padding)
MODEL_PATH = "benchmark_model.h5"  # Model save path
TOKENIZER_PATH = "tokenizer.pkl"  # Tokenizer save path

def open_file_dialog():
    """Open a file dialog to allow the user to select a CSV file."""
    root = Tk()
    root.withdraw()  # Hide the main Tkinter window
    file_path = filedialog.askopenfilename(
        title="Select a CSV File",
        filetypes=[("CSV Files", "*.csv")]
    )
    return file_path

def load_dataset(file_path):
    """Load and preprocess a dataset for benchmarking analysis."""
    try:
        df = pd.read_csv(file_path, encoding='ISO-8859-1', header=None)
    except UnicodeDecodeError:
        print("Error reading the file. Please check the encoding.")
        return None, None, None

    print(f"Loaded dataset from {file_path}")

    df.columns = ['metric', 'ids', 'date', 'flag', 'user', 'text']
    texts = df['text'].astype(str).values
    labels = df['metric'].values

    labels = np.where(labels > 50, 1, 0)  # Example: classify performance metrics into binary categories

    tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    padded_sequences = pad_sequences(sequences, maxlen=MAX_LEN, padding='post', truncating='post')

    return padded_sequences, labels, tokenizer

def create_model(num_classes=1):
    """Create an LSTM model for analyzing benchmarking data."""
    model = Sequential([
        Embedding(input_dim=MAX_WORDS, output_dim=128, input_length=MAX_LEN),
        Bidirectional(LSTM(64, return_sequences=True)),
        GlobalMaxPooling1D(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='sigmoid' if num_classes == 1 else 'softmax')
    ])

    # Use mixed precision compatible optimizer with loss scaling
    opt = Adam(learning_rate=0.001)
    opt = LossScaleOptimizer(opt)
    
    model.compile(optimizer=opt, 
                  loss='binary_crossentropy' if num_classes == 1 else 'categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

def train_model():
    """Train the model using benchmarking datasets."""
    file_path = open_file_dialog()
    if not file_path or not os.path.exists(file_path):
        print("File not found or no file selected.")
        return

    X, y, tokenizer = load_dataset(file_path)
    if X is None:
        return

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42)

    model = create_model()
    print("Training the model...")

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=5,
        batch_size=256,
        verbose=1
    )

    model.save(MODEL_PATH)
    print(f"Model saved at {MODEL_PATH}")

    with open(TOKENIZER_PATH, 'wb') as f:
        pickle.dump(tokenizer, f)
    print(f"Tokenizer saved at {TOKENIZER_PATH}")

def test_model():
    """Test the trained model on benchmarking data."""
    if not os.path.exists(MODEL_PATH) or not os.path.exists(TOKENIZER_PATH):
        print("No trained model found. Please train a model first.")
        return

    print("Loading existing model and tokenizer...")
    model = load_model(MODEL_PATH)
    with open(TOKENIZER_PATH, 'rb') as f:
        tokenizer = pickle.load(f)

    while True:
        choice = input("Enter a performance metric description to analyze (or type 'exit' to quit): ").strip()
        if choice.lower() == 'exit':
            break

        sequence = tokenizer.texts_to_sequences([choice])
        padded_sequence = pad_sequences(sequence, maxlen=MAX_LEN, padding='post', truncating='post')

        prediction = model.predict(padded_sequence)[0][0]
        analysis = "High Performance" if prediction > 0.5 else "Low Performance"
        print(f"Predicted Category: {analysis} (Confidence: {prediction:.2f})")

def main():
    """Main function to run the program."""
    action = input("Do you want to train the model or analyze metrics? (train/analyze): ").strip().lower()
    if action == 'train':
        train_model()
    else:
        test_model()

if __name__ == "__main__":
    main()
