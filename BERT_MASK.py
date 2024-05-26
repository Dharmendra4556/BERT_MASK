import torch
from transformers import BertForMaskedLM, BertTokenizer
import streamlit as st
# Load pre-trained model and tokenizer
model = BertForMaskedLM.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize input sentence
sentence = st.text_input("Enter your question (with[MASK]):")
if st.button("Send", key="send_button"):
    tokens = tokenizer.tokenize('[CLS] ' + sentence + ' [SEP]')
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokens)
    mask_token_index = tokens.index('[MASK]')

    # Create tensors
    ids_tensor = torch.tensor([indexed_tokens])
    mask_tensor = torch.tensor([[1] * len(tokens)])

    # Predict masked token
    with torch.no_grad():
        logits = model(ids_tensor, attention_mask=mask_tensor)[0]

    # Get the predicted token
    predicted_token_id = torch.argmax(logits[0, mask_token_index]).item()
    predicted_token = tokenizer.convert_ids_to_tokens(predicted_token_id)
    # Replace [MASK] with predicted token
    output = sentence.replace('[MASK]', predicted_token)
    st.success(output)