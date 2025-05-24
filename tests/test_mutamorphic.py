import random
import joblib
import numpy as np
from pathlib import Path
from src import config

# Define a synonym dictionary for basic sentiment-related words
SYNONYMS = {
    "okay": ["fine", "all right", "acceptable"],
    "good": ["great", "excellent", "positive"],
    "bad": ["poor", "negative", "unacceptable"],
    "happy": ["glad", "joyful", "pleased"],
    "sad": ["unhappy", "sorrowful", "down"],
}

def synonym_replace(text):
    words = text.split()
    new_words = []
    for w in words:
        lw = w.lower()
        if lw in SYNONYMS:
            new_word = random.choice(SYNONYMS[lw])
            if w[0].isupper():
                new_word = new_word.capitalize()
            new_words.append(new_word)
        else:
            new_words.append(w)
    return ' '.join(new_words)

def generate_mutants(texts):
    return [synonym_replace(text) for text in texts]

def repair_mutant(original_text, orig_label, model, processor, max_attempts=5):
    for attempt in range(max_attempts):
        mutant = synonym_replace(original_text)
        X_mut = processor.transform([mutant]).toarray()
        mut_label = model.predict(X_mut)[0]
        if mut_label == orig_label:
            return mutant, True
    return mutant, False

def metamorphic_test(original_texts, model_path, processor_path):
    model = joblib.load(model_path)
    processor = joblib.load(processor_path)

    X_orig = processor.transform(original_texts).toarray()
    X_mutant_texts = generate_mutants(original_texts)
    X_mut = processor.transform(X_mutant_texts).toarray()

    y_orig_pred = model.predict(X_orig)
    y_mut_pred = model.predict(X_mut)

    mismatches = []
    repaired_samples = []

    for i, (orig_label, mut_label) in enumerate(zip(y_orig_pred, y_mut_pred)):
        if orig_label != mut_label:
            repaired_text, success = repair_mutant(original_texts[i], orig_label, model, processor)
            if success:
                repaired_samples.append((i, repaired_text))
            else:
                mismatches.append(i)

    return mismatches, repaired_samples

def get_latest_model_file(models_dir: Path):
    model_files = list(models_dir.glob("*.joblib"))
    if not model_files:
        raise FileNotFoundError(f"No model files found in {models_dir}")
    model_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
    return model_files[0]


def test_metamorphic_synonym_invariance():
    texts = [
        "This product is okay and I like it.",
        "I feel happy about this service.",
        "The movie was bad and boring."
    ]

    model_path = get_latest_model_file(config.MODELS_DIR)
    processor_path = "vectorizers/preprocessor.joblib"

    mismatches, repaired = metamorphic_test(texts, model_path, processor_path)

    assert len(mismatches) == 0, f"{len(mismatches)} samples could not be repaired"


