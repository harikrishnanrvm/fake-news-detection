from pathlib import Path

# Reproducibility
RANDOM_SEED = 42

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_RAW_DIR = BASE_DIR / "dataset" / "raw"
DATASET_PROCESSED_DIR = BASE_DIR / "dataset" / "processed"
MODELS_DIR = BASE_DIR / "models"
BASELINE_MODEL_DIR = MODELS_DIR / "baseline"
LSTM_MODEL_DIR = MODELS_DIR / "lstm"
REPORT_FIGURES_DIR = BASE_DIR / "report" / "figures"
REPORT_TABLES_DIR = BASE_DIR / "report" / "tables"
EVALUATION_DIR = BASE_DIR / "evaluation"
EXPERIMENTS_LOG_FILE = EVALUATION_DIR / "experiments.csv"

# Staged preprocessing pipeline outputs (see preprocessing/pipeline.py)
STAGE1_COMBINED_FILE = DATASET_PROCESSED_DIR / "01_combined_raw.csv"
STAGE2_CLEANED_FILE = DATASET_PROCESSED_DIR / "02_cleaned.csv"
STAGE3_PREPROCESSED_FILE = DATASET_PROCESSED_DIR / "03_preprocessed.csv"

# Data split
TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

# Tokenizer / sequence settings (used from Phase 5 onward)
MAX_SEQUENCE_LENGTH = 300
VOCAB_SIZE = 20_000
OOV_TOKEN = "<OOV>"
EMBEDDING_DIM = 100

# API validation (used from Phase 7 onward)
MIN_ARTICLE_CHARS = 50
MAX_ARTICLE_CHARS = 20_000

# Baseline model (Phase 4): TF-IDF + Logistic Regression
# Shares VOCAB_SIZE's value for the vocabulary cap so both models are capped
# at a comparable vocabulary size, even though they build it differently.
TFIDF_MAX_FEATURES = VOCAB_SIZE
TFIDF_NGRAM_RANGE = (1, 1)  # unigrams only - keeps "top words" interpretable for the report/viva
LOGISTIC_REGRESSION_MAX_ITER = 1000
TOP_N_FEATURES = 20

# LSTM model (Phase 5)
# VOCAB_SIZE, MAX_SEQUENCE_LENGTH, OOV_TOKEN, EMBEDDING_DIM already defined above -
# shared with the Tokenizer/embedding layer so the whole sequence-preparation
# story (vocab cap, padding length) is configured in exactly one place.
SPATIAL_DROPOUT_RATE = 0.2
LSTM_UNITS = 64
DROPOUT_RATE = 0.3
DENSE_UNITS = 32
LSTM_BATCH_SIZE = 64
LSTM_MAX_EPOCHS = 10
EARLY_STOPPING_PATIENCE = 3
