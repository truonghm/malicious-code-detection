# %%
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchtext

from torchtext.vocab import build_vocab_from_iterator
from torch.nn.utils.rnn import pad_sequence
import pytorch_lightning as pl
from torch.nn.functional import binary_cross_entropy_with_logits, binary_cross_entropy
from torchmetrics import Accuracy, F1Score
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import fasttext

# %%
from sklearn import model_selection

# %%
import os

# %%
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

# %%
from dotenv import load_dotenv
load_dotenv()

os.chdir(os.getenv("PROJECT_ROOT_DIR"))
print(os.getcwd())

# %%
class MODEL_EVAL_METRIC:
    accuracy = "accuracy"
    f1_score = "f1_score"

class Config:
    VOCAB_SIZE = 0
    BATCH_SIZE = 8
    EMB_SIZE = 100
    OUT_SIZE = 2
    NUM_FOLDS = 5
    NUM_EPOCHS = 10
    NUM_WORKERS = 8
    # Whether to update the pretrained embedding weights during training process
    EMB_WT_UPDATE = True
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    MODEL_EVAL_METRIC = MODEL_EVAL_METRIC.accuracy
    FAST_DEV_RUN = False    
    PATIENCE = 6    
    IS_BIDIRECTIONAL = True
    # model hyperparameters
    MODEL_PARAMS = {
        "hidden_size": 141, 
        "num_layers": 2,         
        "drop_out": 0.4258,
        "lr": 0.000366,
        "weight_decay": 0.00001
    }
    X_TEST_PATH = 'data/exp/test_set_token_types_corpus.txt'
    Y_TEST_PATH = 'data/exp/test_set_labels.txt'
    X_TRAIN_PATH = 'data/exp/train_set_token_types_corpus.txt'
    Y_TRAIN_PATH = 'data/exp/train_set_labels.txt' 
    
# For results reproducibility 
# sets seeds for numpy, torch, python.random and PYTHONHASHSEED.
pl.seed_everything(42, workers=True)

# %%
def read_data(path):
	with open(path, "r") as f:
		data = f.readlines()
		vectors = [token.split() for token in data]
		return vectors

# %%
print("Loading data")

X_train = read_data(Config.X_TRAIN_PATH)
y_train_str = np.loadtxt(Config.Y_TRAIN_PATH, dtype=str)
y_train = np.where(y_train_str == 'goodjs', 0.0, 1.0)

X_test = read_data(Config.X_TEST_PATH)
y_test_str = np.loadtxt(Config.Y_TEST_PATH, dtype=str)
y_test = np.where(y_test_str == 'goodjs', 0.0, 1.0)

df_train = pd.DataFrame({'X': X_train, 'y': y_train})
df_train = df_train[df_train["X"].apply(len) != 0]
df_test = pd.DataFrame({'X': X_test, 'y': y_test})
df_test = df_test[df_test["X"].apply(len) != 0]

# %%
def yield_tokens(df):
    for index, row in df.iterrows():
        yield row["X"]

print("Building vocab")
ast_vocab = build_vocab_from_iterator(yield_tokens(df_train), specials=["<unk>", "<pad>"])   
Config.VOCAB_SIZE = len(ast_vocab)

# %%
print("Loading fasttext embeddings")

FASTTEXT_EMB_FILE = "models/fasttext_embeddings.vec"
FASTTEXT_MODEL = "models/fasttext_embeddings.bin"
emb = torchtext.vocab.Vectors(name=FASTTEXT_EMB_FILE, cache="./vector_cache")
fasttext_model = fasttext.load_model(FASTTEXT_MODEL)

# %%
def get_vocab_pt_emb_matrix(text_vocab, emb):
    embedding_matrix = []
    for token in text_vocab.get_itos():
        embedding_matrix.append(emb.get_vecs_by_tokens(token))
    return torch.stack(embedding_matrix)

pt_emb_weights = get_vocab_pt_emb_matrix(ast_vocab, emb).to(Config.DEVICE)
pt_emb_layer = nn.Embedding.from_pretrained(pt_emb_weights).to(Config.DEVICE)

# %%
print("vectorize data")

df_train["vectorized_X"] = df_train["X"].apply(
    lambda row:torch.LongTensor(ast_vocab.lookup_indices(row))
    )

df_test["vectorized_X"] = df_test["X"].apply(
    lambda row:torch.LongTensor(ast_vocab.lookup_indices(row))
    )

# %%
class JavaScriptASTDataset(Dataset):
    def __init__(self, ast_vecs, labels):
        self.ast_vecs = ast_vecs
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        ast_vec = self.ast_vecs[idx]
        label = self.labels[idx]
        # ast_len = len(ast_vec)
        return (ast_vec, label)


# %%
def pad_collate(batch):
    # Each element in the batch is a tuple (data, label)
    # sort the batch (based on tweet word count) in descending order
    sorted_batch = sorted(batch, key=lambda x:x[0].shape[0], reverse=True)
    sequences = [x[0] for x in sorted_batch]
    sequences_padded = pad_sequence(sequences, batch_first=True, padding_value=0)
    # Also need to store the length of each sequence.This is later needed in order to unpad 
    # the sequences
    seq_len = torch.Tensor([len(x) for x in sequences])
    labels = torch.Tensor([x[1] for x in sorted_batch])
    return sequences_padded, seq_len, labels

X_train = df_train["vectorized_X"].to_numpy()
y_train = df_train["y"].to_numpy()
X_valid = df_test["vectorized_X"].to_numpy()
y_valid = df_test["y"].to_numpy()

train_data = JavaScriptASTDataset(X_train, y_train)
train_loader = DataLoader(train_data, shuffle=True, batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS, collate_fn=pad_collate)

# Create DataLoader for test data
test_data = JavaScriptASTDataset(X_valid, y_valid)
test_loader = DataLoader(test_data, shuffle=False, batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS, collate_fn=pad_collate)


# %%
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, dropout, pt_emb_weights):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight.data.copy_(pt_emb_weights)
        self.rnn = nn.LSTM(
            embedding_dim, 
            hidden_dim, 
            num_layers=n_layers, 
            bidirectional=True, 
            dropout=dropout
            )
        
        self.fc = nn.Linear(hidden_dim * 2, 1)
        
        self.dropout = nn.Dropout(dropout)
        # self.act = nn.Sigmoid()

    def forward(self, text, lengths):
        embedded = self.embedding(text)
        embeds_pack = pack_padded_sequence(embedded, lengths.to("cpu"), batch_first=True) 
        output, (hidden, cell) = self.rnn(embeds_pack)
        
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))
        output = self.fc(hidden) 
        # output = self.act(output)
        return output


# %%
model = BiLSTMClassifier(
	vocab_size=Config.VOCAB_SIZE,
	embedding_dim=Config.EMB_SIZE,
	hidden_dim=Config.MODEL_PARAMS["hidden_size"],
	output_dim=Config.OUT_SIZE,
	n_layers=Config.MODEL_PARAMS["num_layers"],
	dropout=Config.MODEL_PARAMS["drop_out"],
	pt_emb_weights=pt_emb_weights
	).to(Config.DEVICE)

# %%
import torch.optim as optim

optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCEWithLogitsLoss()

# %%
import os
import json
import matplotlib.pyplot as plt

# %%
print("Start training")

n_epochs = 5  # Number of epochs; you can adjust this

if not os.path.exists('models'):
    os.makedirs('models')

losses = []
for epoch in range(n_epochs):
    epoch_losses = []
    model.train()
    
    for sequences_padded, seq_len, labels in train_loader:
        sequences_padded, seq_len, labels = sequences_padded.to(Config.DEVICE), seq_len.to(Config.DEVICE), labels.to(Config.DEVICE)
        optimizer.zero_grad()
        
        predictions = model(sequences_padded, seq_len).squeeze(1)

        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()

        epoch_losses.append(loss.item())
    
    avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
    print(f'Epoch: {epoch+1:02}')
    print(f'\tAverage Train Loss: {avg_epoch_loss:.3f}')
    
    # Store average loss for this epoch to losses list
    losses.append(avg_epoch_loss)

    # Save model checkpoint
    checkpoint_path = f'models/model_epoch_{epoch + 1}.pt'
    torch.save(model.state_dict(), checkpoint_path)

plt.plot(losses)
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig('training_loss.png')

# %%
# from sklearn.metrics import accuracy_score

# # Set the model to evaluation mode
# model.eval()

# all_predictions = []
# all_labels = []

# with torch.no_grad():
#     for sequences_padded, seq_len, labels in test_loader:

#         predictions = model(sequences_padded, seq_len).squeeze(1)
#         rounded_preds = torch.round(torch.sigmoid(predictions))

#         all_predictions.extend(rounded_preds.cpu().numpy())
#         all_labels.extend(labels.cpu().numpy())

# accuracy = accuracy_score(all_labels, all_predictions)
# print(f'Accuracy: {accuracy * 100:.2f}%')


