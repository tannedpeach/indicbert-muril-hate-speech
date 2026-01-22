# Hindi-Marathi Hate Speech Detection: IndicBERT v2 vs MuRIL


## Project Overview

This project investigates the performance of two leading multilingual models optimized for Indian languages on the task of hate speech detection:

| Model | Pretraining | Languages | Vocabulary | Key Feature |
|-------|-------------|-----------|------------|-------------|
| **IndicBERT v2** | Indic-only (IndicCorp v2) | 24 Indic | 250K tokens | Optimized for Indian languages |
| **MuRIL** | Translation Language Modeling (TLM) | 17 languages | Standard BERT | Cross-lingual alignment via translation pairs |

### Research Questions

1. **In-Language Performance**: Does IndicBERT's monolingual Indic pretraining outperform MuRIL on Hindi hate speech detection?
2. **Cross-Lingual Transfer**: Does MuRIL's TLM pretraining enable better zero-shot transfer from Hindi to Marathi?
3. **Few-Shot Learning**: Which model learns more efficiently from limited Marathi examples?
4. **Script Similarity**: How does shared Devanagari script affect cross-lingual transfer?

## Methodology

### Training Strategies Compared

| Strategy | Trainable Params | Description |
|----------|------------------|-------------|
| **Normal (Frozen)** | ~1K (0.001%) | Only classification head trained |
| **LoRA** | ~2.6M (0.95%) | Parameter-efficient adapter training |
| **SFT (Full)** | ~110M (100%) | All parameters fine-tuned |

### LoRA Configuration
```python
lora_r = 16           # Rank of low-rank matrices
lora_alpha = 32       # Scaling factor
lora_dropout = 0.1    # Dropout for regularization
target_modules = ["query", "key", "value", "dense"]
```

### Experiments

1. **In-Language Evaluation**: Train on Hindi, evaluate on Hindi test set
2. **Zero-Shot Transfer**: Train on Hindi only, evaluate on Marathi
3. **Few-Shot Learning**: Train on Hindi + k Marathi examples (k = 5, 10, 50)
4. **Training Methods Comparison**: Normal vs LoRA vs SFT

## Datasets

The project uses the **HASOC (Hate Speech and Offensive Content)** benchmark datasets:

| Dataset | Language | Train Samples | Test Samples | Labels |
|---------|----------|---------------|--------------|--------|
| HASOC 2019 (Part 1) | Hindi | 1,318 | - | HOF/NOT |
| HASOC 2019 (Part 2) | Hindi | 4,665 | - | HOF/NOT |
| HASOC 2020 | Hindi | 2,963 | - | HOF/NOT |
| HASOC 2021 | Hindi | 4,594 | - | HOF/NOT |
| HASOC 2021 | Marathi | 1,499 | 375 | HOF/NOT |
| **Combined Hindi** | Hindi | ~13,326 | 663 | HOF/NOT |

**Labels:**
- `NOT`: Non-hateful/offensive content
- `HOF`: Hateful/Offensive content

## Technical Implementation

### Text Preprocessing

Uses `indic-nlp-library` for proper Devanagari script handling:
- Unicode normalization for Hindi and Marathi
- URL and mention removal
- Whitespace cleaning and standardization

### Model Architecture

```
Base Model (IndicBERT/MuRIL)
    └── LoRA Adapters (on attention layers)
        └── Classification Head (2 classes: NOT, HOF)
```

### Training Hyperparameters

```python
max_length = 128          # Token sequence length
batch_size = 16           # Training batch size
learning_rate = 2e-4      # AdamW learning rate
num_epochs = 5            # Training epochs
weight_decay = 0.01       # L2 regularization
warmup_ratio = 0.1        # Learning rate warmup
```

### Evaluation Metrics

- **Macro-F1** (Primary): Official HASOC evaluation metric
- **Accuracy**: Overall classification accuracy
- **Precision/Recall**: Per-class performance analysis
- **Confusion Matrix**: Error pattern visualization

## Project Structure

```
NNDL Project/
├── model_comparison.ipynb            # Main experiment notebook
├── README.md                         # This file
├── Report.pdf          # Final project report
│
└── Datasets/
    ├── hasoc_2020_hi_train.xlsx      # HASOC 2020 Hindi training
    ├── hindi_2019_1.tsv              # HASOC 2019 Hindi (Part 1)
    ├── hindi_2019_2.tsv              # HASOC 2019 Hindi (Part 2)
    ├── hindi_2021.csv                # HASOC 2021 Hindi
    ├── hindi_test_1509.csv           # Hindi test set
    ├── mr_Hasoc2021_train.xlsx       # HASOC 2021 Marathi training
    └── hasoc2021_mr_test-blind-2021.csv  # Marathi test set (blind)
```

## Quick Start

### Prerequisites

```bash
pip install torch transformers datasets accelerate
pip install peft bitsandbytes
pip install indic-nlp-library
pip install scikit-learn seaborn matplotlib pandas
pip install sentencepiece protobuf openpyxl
```

### Running the Experiments

**Option 1: Jupyter Notebook**
```bash
jupyter notebook model_comparison.ipynb
```

**Option 2: Google Colab**
- Upload notebooks to Colab
- Enable GPU runtime (T4 recommended)
- Run cells sequentially

### GPU Requirements

- **Minimum**: 8GB VRAM (with LoRA)
- **Recommended**: 16GB VRAM (Tesla T4, RTX 3090)
- **CPU Mode**: Supported but significantly slower

## Key Features

### 1. Parameter-Efficient Training with LoRA
Reduces trainable parameters by ~100x while maintaining competitive performance.

### 2. Few-Shot Cross-Lingual Transfer
Tests knowledge transfer with minimal target language examples (5, 10, 50 samples).

### 3. Comprehensive Preprocessing
Specialized handling for Devanagari script using `indic-nlp-library`.

### 4. Multi-Dataset Combination
Aggregates data from multiple HASOC editions for robust training.

### 5. Visualization Suite
- Learning curves across shot sizes
- Confusion matrix comparisons
- Performance bar charts

## References

### Models
- [IndicBERT v2](https://huggingface.co/ai4bharat/IndicBERTv2-MLM-Sam-TLM) - AI4Bharat
- [MuRIL](https://huggingface.co/google/muril-base-cased) - Google Research India

### Datasets
- [HASOC 2019](https://hasocfire.github.io/hasoc/2019/index.html) - Hate Speech and Offensive Content Identification
- [HASOC 2020](https://hasocfire.github.io/hasoc/2020/index.html)
- [HASOC 2021](https://hasocfire.github.io/hasoc/2021/index.html)

### Libraries
- [LoRA (PEFT)](https://github.com/huggingface/peft) - Parameter-Efficient Fine-Tuning
- [indic-nlp-library](https://github.com/anoopkunchukuttan/indic_nlp_library) - Indic NLP Tools

