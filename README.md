# Depression Detection in Tamil & Malayalam Speech - Audio Classification

## Overview

This Jupyter Notebook implements a comprehensive audio classification pipeline for detecting depression from speech in **Tamil** and **Malayalam** languages. The project uses pre-trained transformers-based models with ensemble techniques to classify audio samples as either "Depressed" or "Non-depressed" based on acoustic features.

---

## Table of Contents

1. [Project Goals](#project-goals)
2. [Technologies & Libraries Used](#technologies--libraries-used)
3. [Step-by-Step Guide](#step-by-step-guide)
4. [Dataset Structure](#dataset-structure)
5. [Model Architecture](#model-architecture)
6. [Evaluation Methodology](#evaluation-methodology)
7. [Key Results](#key-results)
8. [Usage Instructions](#usage-instructions)

---

## Project Goals

- **Detect depression** from audio speech in Tamil and Malayalam languages
- **Compare multiple backbone models** (HuBERT, XLS-R, Whisper) for audio classification
- **Conduct ablation studies** to evaluate individual models vs. ensemble approaches
- **Ensure speaker-disjoint train/test splits** to prevent data leakage
- **Validate predictions** against ground-truth CSV files to measure accuracy

---

## Technologies & Libraries Used

### Core Libraries

| Library | Purpose | Version |
|---------|---------|---------|
| **PyTorch** | Deep learning framework for model training | Latest |
| **Transformers** | Pre-trained audio models (HuBERT, XLS-R, Whisper) | 4.40.2+ |
| **Librosa** | Audio signal processing and loading | Latest |
| **Datasets** | Hugging Face dataset management library | Latest |
| **Scikit-learn** | Metrics & model evaluation (F1, precision, recall) | Latest |
| **Pandas** | Data manipulation and CSV handling | Latest |
| **NumPy** | Numerical computations | Latest |

### Pre-trained Models

1. **HuBERT** (`facebook/hubert-large-ls960-ft`)
   - Self-supervised speech representation model
   - 24 transformer layers, 1024-dim embeddings
   - Fine-tuned on LibriSpeech with CTC loss

2. **XLS-R** (`facebook/wav2vec2-large-xlsr-53`)
   - Cross-lingual speech representation learning
   - Trained on 53 languages
   - Effective for low-resource languages like Tamil & Malayalam

3. **Whisper** (`openai/whisper-small`)
   - OpenAI's speech recognition model
   - Trained on multilingual audio
   - Robust to various acoustic conditions

### GPU Acceleration

- **CUDA** for GPU computing (PyTorch with CUDA support)
- **gradient_checkpointing** to reduce memory usage
- **fp16 (Mixed Precision)** training for faster convergence

---

## Step-by-Step Guide

### Step 1: Environment Setup & Dependencies Installation

```python
# Install required packages
!pip install -q transformers datasets librosa torchaudio accelerate soundfile

# Verify GPU availability
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)  # Should output 'cuda' for GPU
```

**Purpose:** Install audio processing and transformer libraries; verify GPU availability for efficient training.

---

### Step 2: Data Preparation & Loading

#### 2.1 Define Data Paths

```python
BASE_PATH = "/kaggle/input/tamil-malayalam-deppression-audio"

# Resolve directory paths with fallback options
TAMIL_DEP = "/path/to/Tamil/Depressed/Train_set"
TAMIL_NON = "/path/to/Tamil/Non-depressed/Train_set"
MAL_DEP = "/path/to/Malayalam/Depressed/Train_set"
MAL_NON = "/path/to/Malayalam/Non-depressed/Train_set"
```

**Purpose:** Specifies locations of training data for both languages and depression classes.

#### 2.2 Audio Loading & Preprocessing

```python
def load_audio(path):
    # Load audio at 16 kHz sampling rate
    audio, _ = librosa.load(path, sr=TARGET_SR, mono=True)
    
    # Enforce maximum length (5 seconds = 80,000 samples)
    if len(audio) > MAX_LEN:
        audio = audio[:MAX_LEN]
    else:
        audio = np.pad(audio, (0, MAX_LEN - len(audio)))
    
    return audio
```

**Configuration:**
- `TARGET_SR = 16000` Hz (standard for speech models)
- `MAX_LEN = 5 * 16000 = 80,000` samples (5 seconds maximum)

**Purpose:** Standardizes all audio to consistent length and sampling rate.

#### 2.3 Extract Speaker IDs

```python
def get_speaker_id(filename):
    # Extract speaker identifier from filename
    # Example: "speaker123_audio.wav" → "speaker123"
    base = os.path.splitext(os.path.basename(filename))[0]
    for sep in ("_", "-"):
        if sep in base:
            return base.split(sep)[0]
    return base
```

**Purpose:** Extract speaker identifiers for speaker-disjoint evaluation.

#### 2.4 Build Training DataFrames

```python
def build_train_df(dep_path, nondep_path):
    rows = []
    
    # Load depressed samples (label=1)
    for f in os.listdir(dep_path):
        if f.endswith(".wav"):
            rows.append({
                "path": os.path.join(dep_path, f),
                "label": 1,  # Depressed
                "speaker": get_speaker_id(f)
            })
    
    # Load non-depressed samples (label=0)
    for f in os.listdir(nondep_path):
        if f.endswith(".wav"):
            rows.append({
                "path": os.path.join(nondep_path, f),
                "label": 0,  # Non-depressed
                "speaker": get_speaker_id(f)
            })
    
    return pd.DataFrame(rows, columns=["path", "label", "speaker"])

tamil_train_df = build_train_df(TAMIL_DEP, TAMIL_NON)
mal_train_df = build_train_df(MAL_DEP, MAL_NON)
```

**Output:** DataFrames with columns: `path`, `label` (0/1), `speaker`

---

### Step 3: Load Pre-trained Models & Processors

#### 3.1 Initialize HuBERT Model

```python
hubert_processor = Wav2Vec2Processor.from_pretrained(
    "facebook/hubert-large-ls960-ft"
)
hubert_model = HubertForSequenceClassification.from_pretrained(
    "facebook/hubert-large-ls960-ft",
    num_labels=2  # Binary classification: depressed vs non-depressed
).to(DEVICE)
```

#### 3.2 Initialize XLS-R Model (Cross-lingual)

```python
xlsr_processor = Wav2Vec2FeatureExtractor.from_pretrained(
    "facebook/wav2vec2-large-xlsr-53"
)
xlsr_model = Wav2Vec2ForSequenceClassification.from_pretrained(
    "facebook/wav2vec2-large-xlsr-53",
    num_labels=2
).to(DEVICE)
```

#### 3.3 Initialize Whisper Model

```python
whisper_processor = AutoProcessor.from_pretrained("openai/whisper-small")
whisper_model = WhisperForAudioClassification.from_pretrained(
    "openai/whisper-small",
    num_labels=2
).to(DEVICE)
```

**Purpose:** Load three distinct pre-trained models for ensemble training.

---

### Step 4: Dataset Conversion to Hugging Face Format

```python
def to_hf_dataset(df, processor):
    """Convert pandas DataFrame to Hugging Face Dataset"""
    ds = Dataset.from_pandas(df.reset_index(drop=True))
    
    def preprocess(batch):
        # Load and process audio
        audio = [load_audio(p) for p in batch["path"]]
        inputs = processor(audio, sampling_rate=TARGET_SR)
        
        return {
            "input_values": inputs.input_values,
            "label": batch["label"]
        }
    
    # Apply preprocessing in batches
    ds = ds.map(preprocess, batched=True, remove_columns=ds.column_names)
    
    # Convert to PyTorch format
    ds.set_format(type="torch", columns=["input_values", "label"])
    
    return ds
```

**Purpose:** Converts raw audio paths to tensors compatible with PyTorch models.

---

### Step 5: Create Data Splits

#### 5.1 Standard 80/20 Split

```python
from datasets import ClassLabel

def build_splits(df, processor):
    full_ds = to_hf_dataset(df, processor)
    full_ds = full_ds.cast_column("label", ClassLabel(num_classes=2))
    split_ds = full_ds.train_test_split(
        test_size=0.2,
        stratify_by_column="label"  # Balance class distribution
    )
    return split_ds["train"], split_ds["test"]

tamil_train_ds, tamil_eval_ds = build_splits(tamil_train_df, hubert_processor)
mal_train_ds, mal_eval_ds = build_splits(mal_train_df, hubert_processor)
```

#### 5.2 Speaker-Disjoint Split (Prevents Data Leakage)

```python
def build_speaker_disjoint_splits(df, processor, test_size=0.2):
    """Ensures no speaker appears in both train and test"""
    from sklearn.model_selection import train_test_split
    
    speakers = df["speaker"].unique()
    train_speakers, test_speakers = train_test_split(
        speakers,
        test_size=test_size,
        random_state=42
    )
    
    train_df = df[df["speaker"].isin(train_speakers)].reset_index(drop=True)
    test_df = df[df["speaker"].isin(test_speakers)].reset_index(drop=True)
    
    train_ds = to_hf_dataset(train_df, processor)
    test_ds = to_hf_dataset(test_df, processor)
    
    return train_ds, test_ds
```

**Advantage:** Prevents model from memorizing speaker characteristics instead of depression cues.

---

### Step 6: Model Training with Gradient Accumulation

```python
def train_hubert(language, train_ds, eval_ds, processor):
    """Train HuBERT model with memory optimization"""
    
    model = HubertForSequenceClassification.from_pretrained(
        "facebook/hubert-large-ls960-ft",
        num_labels=2
    )
    
    # Freeze backbone, train only classifier
    for param in model.hubert.parameters():
        param.requires_grad = False
    
    # Make classifier trainable
    for param in model.classifier.parameters():
        param.requires_grad = True
    
    # Enable gradient checkpointing to save memory
    model.gradient_checkpointing_enable()
    model.to(DEVICE)
    
    # Training arguments
    args = TrainingArguments(
        output_dir=f"./dddl/{language}",
        learning_rate=1e-4,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,  # Effective batch size = 8
        num_train_epochs=20,
        fp16=True,  # Mixed precision training
        
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,  # Keep only best checkpoint
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        
        logging_strategy="epoch",
        report_to="none"
    )
    
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    
    trainer.train()
    trainer.save_model(f"/kaggle/working/hubert_{language}")
    
    # Clean up GPU memory
    del model, trainer
    torch.cuda.empty_cache()
    
    return f"/kaggle/working/hubert_{language}"
```

**Key Training Parameters:**
- **Learning Rate:** 1e-4 (conservative for fine-tuning)
- **Batch Size:** 1 (memory constraint for large models)
- **Gradient Accumulation:** 8 steps (simulates batch size 8)
- **Mixed Precision (fp16):** Reduces memory & speeds up training
- **Early Stopping:** Stops if validation metric doesn't improve for 3 epochs
- **Metric:** Macro F1-score (balanced for imbalanced datasets)

---

### Step 7: Evaluation & Metrics Computation

```python
def compute_metrics(eval_pred):
    """Calculate evaluation metrics"""
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    
    macro_f1 = f1_score(labels, preds, average="macro")
    precision = precision_score(labels, preds, average="macro")
    recall = recall_score(labels, preds, average="macro")
    
    return {
        "macro_f1": macro_f1,
        "precision": precision,
        "recall": recall
    }
```

**Metrics Explanation:**
- **Macro F1:** Average F1 across both classes (accounts for imbalance)
- **Precision:** True positives / (True positives + False positives)
- **Recall:** True positives / (True positives + False negatives)

---

### Step 8: Speaker-Disjoint Evaluation

```python
def evaluate_on_split(model, test_ds, processor):
    """Evaluate model on held-out test speakers"""
    model.eval()
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for i in range(len(test_ds)):
            item = test_ds[i]
            input_vals = item["input_values"].unsqueeze(0).to(DEVICE)
            label = item["label"]
            
            logits = model(input_vals).logits
            pred = torch.argmax(logits, dim=-1).item()
            
            all_preds.append(pred)
            all_labels.append(label)
    
    metrics = {
        "macro_f1": f1_score(all_labels, all_preds, average="macro"),
        "precision": precision_score(all_labels, all_preds, average="macro"),
        "recall": recall_score(all_labels, all_preds, average="macro")
    }
    
    return metrics
```

---

### Step 9: Ablation Study - Individual Models vs Ensemble

```python
def eval_ensemble_on_df(hubert_model, xlsr_model, whisper_model, eval_df):
    """Ensemble all three models with equal weighting"""
    all_preds = []
    
    for _, row in eval_df.iterrows():
        audio = load_audio(row["path"])
        
        # Get logits from each model
        h_logits = hubert_model(process_audio(audio)).logits
        x_logits = xlsr_model(process_audio(audio)).logits
        w_logits = whisper_model(process_audio(audio)).logits
        
        # Average ensemble logits
        ensemble_logits = (h_logits + x_logits + w_logits) / 3
        pred = torch.argmax(ensemble_logits, dim=-1).item()
        
        all_preds.append(pred)
    
    return all_preds
```

**Ensemble Strategy:**
- Simple averaging of logits from 3 models
- No learnable weights (can be extended with weighted ensemble)
- Improves robustness compared to individual models

---

### Step 10: Generate Predictions & CSV Output

```python
def predict_test_hubert(test_dir, model):
    """Predict on test directory and save as CSV"""
    model.eval()
    rows = []
    
    files = sorted([f for f in os.listdir(test_dir) if f.endswith(".wav")])
    
    with torch.no_grad():
        for f in files:
            audio = load_audio(f)
            logits = model(process_audio(audio)).logits
            pred = torch.argmax(logits, dim=-1).item()
            
            label = "Depressed" if pred == 1 else "Non-depressed"
            rows.append({"file": os.path.basename(f), "label": label})
    
    return pd.DataFrame(rows)

# Generate predictions
tamil_pred = predict_test_hubert(TAMIL_TEST, hubert_model_tamil)
mal_pred = predict_test_hubert(MAL_TEST, hubert_model_malayalam)

# Save CSVs
tamil_pred.to_csv("VEL_Tamil_HuBERT.csv", index=False)
mal_pred.to_csv("VEL_Malayalam_HuBERT.csv", index=False)
```

---

### Step 11: Comparison with Ground Truth

```python
def compare_with_gt(gt_csv, pred_csv, language_name):
    """Compare predictions with ground truth and compute metrics"""
    gt_df = pd.read_csv(gt_csv)
    pred_df = pd.read_csv(pred_csv)
    
    # Merge on filename
    merged = gt_df.merge(pred_df, on="file")
    
    # Compute accuracy and F1-score
    accuracy = accuracy_score(merged["gt_label"], merged["pred_label"])
    f1 = f1_score(merged["gt_label"], merged["pred_label"], average="macro")
    
    print(f"{language_name} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
    
    return accuracy, f1
```

---

## Dataset Structure

```
tamil-malayalam-depression-audio/
├── Tamil/
│   ├── Depressed/
│   │   └── Train_set/
│   │       ├── speaker001_001.wav
│   │       ├── speaker001_002.wav
│   │       └── ...
│   └── Non-depressed/
│       └── Train_set/
│           ├── speaker101_001.wav
│           └── ...
├── Malayalam/
│   ├── Depressed/
│   │   └── Train_set/
│   │       └── ...
│   └── Non-depressed/
│       └── Train_set/
│           └── ...
├── Test-set-tamil/
│   ├── test_001.wav
│   └── ...
└── Test_set_mal/
    ├── test_001.wav
    └── ...
```

**Class Distribution:** Approximately balanced between depressed/non-depressed for both languages.

---

## Model Architecture

### HuBERT (Primary Model)

```
Input Audio (16kHz, max 5 sec)
    ↓
Feature Extractor (Conv1D)
    ↓
HuBERT Encoder (24 blocks of self-attention)
    ↓
Pooling Layer (mean pooling over time)
    ↓
Classification Head (Linear, 2 outputs)
    ↓
Softmax → Probabilities [P(Non-depressed), P(Depressed)]
```

### XLS-R & Whisper

- Similar architecture with multilingual pre-training
- XLS-R: trained on 53 languages (good generalization)
- Whisper: trained on diverse audio conditions (robust)

### Ensemble

```
Audio Input
    ├─→ HuBERT → logits₁
    ├─→ XLS-R → logits₂
    └─→ Whisper → logits₃
         ↓
    Average: (logits₁ + logits₂ + logits₃) / 3
         ↓
    Final Prediction
```

---

## Evaluation Methodology

### 1. Speaker-Disjoint Evaluation
- **Train speakers:** First 80% of unique speakers
- **Test speakers:** Last 20% of unique speakers
- **Prevents:** Model learning speaker-specific patterns instead of depression

### 2. Stratified Split
- Maintains class balance in train/test
- Prevents biased evaluations on imbalanced data

### 3. Metrics Computed
| Metric | Formula | Use Case |
|--------|---------|----------|
| **Accuracy** | (TP + TN) / Total | Overall correctness |
| **Macro F1** | avg(F1_depressed, F1_non-depressed) | Imbalanced datasets |
| **Precision** | TP / (TP + FP) | False positive rate |
| **Recall** | TP / (TP + FN) | False negative rate |

### 4. Ground Truth Validation
- Predictions compared with uploaded GT CSV files
- Cross-validation with official test labels
- Compute final accuracy/F1 on official test set

---

## Key Results

### Speaker-Disjoint Evaluation
- **Tamil (HuBERT):** Macro-F1 ≈ [Value depends on data]
- **Malayalam (HuBERT):** Macro-F1 ≈ [Value depends on data]

### Ablation Study (Stratified Split)
| Model | Accuracy | Macro-F1 | Precision | Recall |
|-------|----------|----------|-----------|--------|
| HuBERT (Tamil) | [X] | [X] | [X] | [X] |
| Ensemble (Tamil) | [X] | [X] | [X] | [X] |
| HuBERT (Malayalam) | [Y] | [Y] | [Y] | [Y] |
| Ensemble (Malayalam) | [Y] | [Y] | [Y] | [Y] |

### Predictions Generated
- `VEL_Tamil_HuBERT.csv` - Individual model predictions
- `VEL_Tamil_Ensemble.csv` - Ensemble predictions
- `VEL_Malayalam_HuBERT.csv` - Individual model predictions
- `VEL_Malayalam_Ensemble.csv` - Ensemble predictions

---

## Usage Instructions

### 1. Run the Full Pipeline

```python
# Execute cells in order (1-53)
# Kaggle environment recommended for GPU support

# Key checkpoints:
# - After Step 2: Data loading confirmed
# - After Step 6: Models trained (check GPU memory)
# - After Step 8: Speaker-disjoint evaluation completed
# - After Step 11: CSV files generated
```

### 2. Custom Dataset

To use your own dataset:

```python
BASE_PATH = "/path/to/your/dataset"
TAMIL_DEP = "/path/to/tamil/depressed"
TAMIL_NON = "/path/to/tamil/non-depressed"
# ... (repeat for Malayalam)

# Then run remaining cells
```

### 3. Modify Model Configuration

```python
# Change learning rate
learning_rate = 5e-5

# Change number of epochs
num_train_epochs = 30

# Unfreeze more layers
unfreeze_last_n = 6  # Instead of 0

# Adjust ensemble weights
ensemble_logits = (0.5*h + 0.3*x + 0.2*w)  # Custom weights
```

---

## Memory & Performance Optimization

| Technique | Benefit | Usage |
|-----------|---------|-------|
| **Gradient Checkpointing** | -30% GPU memory | `model.gradient_checkpointing_enable()` |
| **Gradient Accumulation** | Larger effective batch size | `gradient_accumulation_steps=8` |
| **Mixed Precision (fp16)** | 2x faster training | `fp16=True` |
| **Memory Clearing** | Prevent OOM errors | `torch.cuda.empty_cache()` |

---

## References

- **HuBERT:** [Facebook Research](https://arxiv.org/abs/2106.07447)
- **XLS-R:** [Cross-lingual Speech Representations](https://arxiv.org/abs/2111.02027)
- **Whisper:** [OpenAI Speech Recognition](https://arxiv.org/abs/2212.04356)
- **Transformers Library:** [Hugging Face](https://huggingface.co/transformers/)

---

## Notes

- **GPU Required:** Models are too large for CPU training (~1.3GB each)
- **Inference Speed:** ~0.5-1s per audio sample on GPU
- **Best Practices:** Always use speaker-disjoint splits for depression detection
- **Data Privacy:** Audio data should be handled securely per HIPAA/GDPR compliance

---

## Authors & License

- **Team:** CUET_Neural_Networks
- **Authors:** Shuva Dey,Abir Dey 
---

**Last Updated:** March 2026
