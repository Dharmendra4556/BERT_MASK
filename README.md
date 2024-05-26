# BERT Masked Language Model with Streamlit

This repository contains a simple web application built with Streamlit that uses a pre-trained BERT model to predict masked tokens in a given sentence. The application allows users to input a sentence with a masked word and returns the most likely prediction for that masked word.

## Features

- Utilizes the `bert-base-uncased` pre-trained model from the Hugging Face Transformers library.
- Allows user input for sentences with a masked token (`[MASK]`).
- Displays the predicted token to replace the `[MASK]`.
