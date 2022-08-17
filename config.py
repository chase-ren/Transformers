import transformers

# Parameters
MAX_LEN = 512
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 4
EPOCHS = 10

# Path to BERT model files
BERT_PATH = "../input/bert_base_uncased/"

# Saving model
MODEL_PATH = "model.bin"

# Training file
TRAINING_FILE = "../input/file.csv"

# Define the tokenizer
TOKENIZER = transformers.BertTokenizer.from_pretrained(
    BERT_PATH,
    do_lower_case=True
)
