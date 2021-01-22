# valprevstan
Valprevstan is a collection of scripts used during an internship project at the AUMC Amsterdam combined with a thesis at the VU Amsterdam. These scripts are used to construct an SVM
that categorizes whether sentences in medical patient notes are describing a fall-incident or not. This will then be used for a fall-prevention project.

## Installation

```bash
pip install os
pip install xml
pip install nltk
pip install pickle
pip install fasttext
pip install pandas
pip install sklearn
pip install numpy
pip install imblearn
```

## Usage
Right now, all paths and configurations are hardcoded (and indicated as such in the comments of the code). In the future, command line arguments may be added.

To use this code, first run sample_training_data.py to create a dataset to train the word embedding model on. 
--Possible configurations: paths. 
[Optional]: run create_prep_dataset_emb.py if word embedding model is to be trained on preprocessed data.
--Possible configurations: paths.
Then, run word_emb.py to train the word embedding model. This takes very long. 
--Possible configurations: parameters model, path inputfile. 
Afterwards, run collect_all_annotations.py to map xml-files to the textfiles. This also maps the sentences to the word embeddings. 
--Possible configurations: paths.
Finally, SVMs can be trained. First run SVM_train.py. 
--Possible configurations: paths, hyperparameters model, rebalancing techniques, preprocessing steps, the use of a gridsearch including its parameters.
Then, run SVM2_train.py. 
--Possible configurations: paths, hyperparameters model, rebalancing techniques, preprocessing steps, the use of a gridsearch including its parameters.

Now, SVM_predict.py and SVM2_predict.py can be used to categorize sentences in medical notes based on whether they describe fall-incidents. 
Possible configurations: paths. 
Important! input file needs to be run through the data_into_df.py script first, which takes a dictionary as input: {filename: [[sentence,sentence],[id_of_sentence_with tag],[label]}