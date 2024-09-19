from transformers import BertTokenizer, BertModel
import torch
import faiss
import numpy as np
import os
import glob

import openai
from langchain_openai import AzureChatOpenAI

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
        # Add embeddings to the index
        index.add(embeddings)

# Save the index to a file
faiss.write_index(index, 'vector_index.faiss')

# Load the FAISS index from the file
index = faiss.read_index('vector_index.faiss')

# Get the total number of vectors in the index
num_vectors = index.ntotal

# Retrieve all vectors from the index
vectors = []
for i in range(num_vectors):
    vector = index.reconstruct(i)
    vectors.append(vector)
    
# Convert the list of vectors to a NumPy array
vectors = np.array(vectors)

def generate_response(vectors, prompt):
    # Convert vectors to input format
    vector_text = " ".join([str(v) for v in vectors.tolist()])
    combined_input = vector_text + " " + prompt

    # Initialize AzureChatOpenAI
    llm = AzureChatOpenAI(openai_api_version="2023-03-15-preview",azure_deployment="TheInformant",api_key="", azure_endpoint="https://drainthebrain.openai.azure.com")
    # Generate response using AzureChatOpenAI
    response = llm(combined_input, max_tokens=100, temperature=0.7)
    return response

# Example usage
prompt = "The text before this string"
response = generate_response(vectors, prompt)
print(response)