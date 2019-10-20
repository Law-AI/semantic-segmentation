# Semantic Segmentation of Indian Supreme Court Case Documents

## Introduction
Every sentence in a court case document can be assigned a rhetorical (semantic) role, such as 'Arguments', 'Facts', 'Ruling by Present Court', etc. A single document is considered as a single training example, represented as a sequence of sentences. We use a deep neural model, 'Hierarchical BiLSTM CRF' to semantically segment court documents. (https://link_to_paper_here)

## Requirements
python = 3.7.3
pytorch = 1.1.0
sklearn = 0.21.3
numpy = 1.17.2
sent2vec (https://github.com/epfml/sent2vec)

## Codes
  "model/submodels.py":         Contains codes for submodels that is used for constructing the top-level architecture
  "model/Hier_BiLSTM_CRF.py":   Contains code for the top-level architecture
  "prepare_data.py":            Functions to prepare numericalized data from raw text files
  "train.py":                   Functions for training, validation and learning
  "run.py":                     For reproducing results in the paper
  "infer.py":                   For using a trained model to infer labels for unannotated documents

## Training
### Input Data format
For training and validation, data is placed inside "data/text" folder. Each document is represented as an individual text file, with one sentence per line. The format is: 
  text <TAB> label
  
If you wish to use pretrained embeddings variant of the model, data is placed inside "data/pretrained_embeddings" folder. Each document is represented as an individual text file, with one sentence per line. The format is: 
  emb_f1 <SPACE> emb_f2 <SPACE> ... <SPACE> emb_f200 <TAB> label  (For 200 dimensional sentence embeddings)
  
"categories.txt" contains the category information of documents in the format:
  category_name <TAB> doc <SPACE> doc <SPACE> ...
  
### Usage
To run experiments with default setup, use: 
  python run.py                                                                 (no pretrained variant)
  python run.py --pretrained True --data_path data/pretrained_embeddings/       (pretrained variant)
Constants, hyper parameters and path to data files can be provided as switches along with the previous command, to know more use: 
  python run.py -h
To see default values, check out "run.py"

By default, the model employs 5 fold cross-validation on a total of 50 documents, where folds are manually constructed to have balanced category distribution across each fold.

### Output Data format
All output data will be found inside "saved" folder. This contains:
  model_state_fn.tar:  fn is the validation fold number. This contains the architecture and model state which achieved highest macro-f1 on validation.
  data_state_fn.json:   Contains predictions, true labels, loss and training index for each document in the validation fold.
  word2idx.json and tag2idx.json: Needed for inference
  
## Inference
### Input Data format
Un-annotated data is to be placed inside "infer/data" folder. Each document should be represented as an individual text file, containing one sentence per line.
For inference, we need a trained Hier-BiLSTM-CRF model. For this, place model_state.tar, word2idx.json and tag2idx.json (which were obtained after the training process) inside "infer" folder.
For pretrained variant, we also need to place a trained sent2vec model inside "infer" folder. (https://link_to_sent2vec_model_here)

### Usage
To infer with default setup, use:
  python infer.py                       (no pretrained variant)
  python infer.py --pretrained True     (pretrained variant)
Constants, hyper parameters and path to data files can be provided as switches along with the previous command, to know more use: 
  python infer.py -h
To see default values, check out "infer.py"

### Output Data format
Output will be saved in "infer/predictions.txt", which has the format:
  document_filename <TAB> label_sent1 <COMMA> label_sent2 <COMMA> ... <COMMA> label_sentN     (N sentences in this document)
  
# Notes
1.  Make sure to set the switch --device cpu (or change the default value) if cuda is not available.
2.  Remove the blank "__init__.py" files before running experiments.
