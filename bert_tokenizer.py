from transformers import BertTokenizer, BertModel
import torch
import faiss
import numpy as np
import os
import glob


# Load pre-trained model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Specify the folder path
folder_path = 'transcripts'

# Get a list of all files in the folder
files = glob.glob(os.path.join(folder_path, '*'))

d = 768
index = faiss.IndexFlatL2(d)
# Get a list of all files in the folder
# Iterate through the files and read their content
for file in files:
    with open(file, 'r',encoding='utf-8') as f:
        content = f.read()
        # Tokenize the text
        inputs = tokenizer(content, return_tensors='pt', max_length=512, truncation=True)
        # Generate embeddings
        with torch.no_grad():
            outputs = model(**inputs)\
        # Get the embeddings for the [CLS] token
        embeddings = outputs.last_hidden_state[:, 0, :].numpy()
        print("shape:" , embeddings.shape)
        # Add embeddings to the index
        index.add(embeddings)

# Save the index to a file
faiss.write_index(index, 'vector_index.faiss')