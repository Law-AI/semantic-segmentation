# Semantic Segmentation of Indian Supreme Court Case Documents

## Introduction
This is the repository for the paper titled "Identification of Rhetorical Roles of Sentences in Indian Legal Judgments" which is to be presented at the <a href="https://jurix2019.oeg-upm.net/index.html">International Conference on Legal Knowledge and Information Systems (JURIX) 2019</a>.

Every sentence in a court case document can be assigned a rhetorical (semantic) role, such as 'Arguments', 'Facts', 'Ruling by Present Court', etc. The task of assigning rhetorical roles to individual sentences in a document is known as semantic segmentation. We have developed a deep neural model (Hierarchical BiLSTM CRF) for automatic segmentation of Indian court case documents. A single document is represented as a sequence of sentences. We have used 7 labels for this task: Arguments, Precedent, Statutes, Facts, Ratio Decidendi, Ruling of Lower Court, Ruling of Present Court. 

We make available

(1) a set of 50 court case documents judged in the Supreme Court of India, where each sentence has been annotated with its rhetorical role by law student (see the paper for details)

(2) the implementation of our best performing model (Hierarchical BiLSTM CRF)

## Citation
If you use this dataset or the codes, please refer to the following paper:
```
  @inproceedings{bhattacharya-jurix19,
   author = {Bhattacharya, Paheli and Paul, Shounak and Ghosh, Kripabandhu and Ghosh, Saptarshi and Wyner, Adam},
   title = {{Identification of Rhetorical Roles of Sentences in Indian Legal Judgments}},
   booktitle = {{Proceedings of the 32nd International Conference on Legal Knowledge and Information Systems (JURIX)}},
   year = {2019}
  }
```
## Requirements
- python = 3.7.3
- pytorch = 1.1.0
- sklearn = 0.21.3
- numpy = 1.17.2
- <a href="https://github.com/epfml/sent2vec">sent2vec</a>

## Codes
- _model/submodels.py_ :        Contains codes for submodels that is used for constructing the top-level architecture  
- _model/Hier_BiLSTM_CRF.py_ :  Contains code for the top-level architecture
- _prepare_data.py_ :           Functions to prepare numericalized data from raw text files
- _train.py_ :                  Functions for training, validation and learning
- _run.py_ :                    For reproducing results in the paper
- _infer.py_ :                  For using a trained model to infer labels for unannotated documents

## Training
For training a model on an annotated dataset

### Input Data format
For training and validation, data is placed inside "data/text" folder. Each document is represented as an individual text file, with one sentence per line. The format is: 
  ```
  text <TAB> label
  ```
If you wish to use pretrained embeddings variant of the model, data is placed inside "data/pretrained_embeddings" folder. Each document is represented as an individual text file, with one sentence per line. The format is: 
  ```
  emb_f1 <SPACE> emb_f2 <SPACE> ... <SPACE> emb_f200 <TAB> label  (For 200 dimensional sentence embeddings)
  ```
"categories.txt" contains the category information of documents in the format:
  ```
  category_name <TAB> doc <SPACE> doc <SPACE> ...
  ```
### Usage
To run experiments with default setup, use: 
  ```
  python run.py                                                                 (no pretrained variant)
  python run.py --pretrained True --data_path data/pretrained_embeddings/       (pretrained variant)
  ```
Constants, hyper parameters and path to data files can be provided as switches along with the previous command, to know more use: 
  ```
  python run.py -h
  ```
To see default values, check out "run.py"

By default, the model employs 5 fold cross-validation on a total of 50 documents, where folds are manually constructed to have balanced category distribution across each fold.

### Output Data format
All output data will be found inside "saved" folder. This contains:
- _model_state_fn.tar_ :  fn is the validation fold number. This contains the architecture and model state which achieved highest macro-f1 on validation. 
- _data_state_fn.json_ : Contains predictions, true labels, loss and training index for each document in the validation fold.  
- _word2idx.json_ and _tag2idx.json_ :  Needed for inference
  
## Inference
For using a trained model to automatically annotate documents

### Input Data format
Un-annotated data is to be placed inside "infer/data" folder. Each document should be represented as an individual text file, containing one sentence per line.

For inference, we need a trained Hier-BiLSTM-CRF model. For this, place model_state.tar, word2idx.json and tag2idx.json (which were obtained after the training process) inside "infer" folder.

For pretrained variant, we also need to place a trained sent2vec model inside "infer" folder. 
You can download a sent2vec model pretrained on Indian Supreme Court case documents <a href="http://cse.iitkgp.ac.in/~saptarshi/models/sent2vec.bin"> here </a> (binary file of size more than 2 GB).

### Usage
To infer with default setup, use:
  ```
  python infer.py                       (no pretrained variant)
  python infer.py --pretrained True     (pretrained variant)
  ```
Constants, hyper parameters and path to data files can be provided as switches along with the previous command, to know more use: 
  ```
  python infer.py -h
  ```
To see default values, check out "infer.py"

### Output Data format
Output will be saved in "infer/predictions.txt", which has the format:
  ```
  document_filename <TAB> label_sent1 <COMMA> label_sent2 <COMMA> ... <COMMA> label_sentN     (N sentences in this document)
  ```
# Notes
1.  Make sure to set the switch --device cpu (or change the default value) if cuda is not available.
2.  Remove the blank "__init__.py" files before running experiments.
