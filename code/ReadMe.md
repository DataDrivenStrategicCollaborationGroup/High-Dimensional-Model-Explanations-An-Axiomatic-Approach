This is the code for the experiments in the paper 

## System requirements and setup
The code is written in python 3 and was originally run on a machine with d 40 Intel(R) Xeon(R) Gold 5115 CPU @ 2.40GHz. To reproduce the experiments including the BERT model, the sytem needs to be able run the BERT model (i.e., GPUs are required). 

### Libraries
Running the file `setup.sh` will install all necessary libraries assuming Python 3 and pip are installed. For the requirements of BERT see [here](https://github.com/artemisart/bert-sentiment-IMDB).

### Datasets
We included the adult dataset in this code as it is small. The IMDB dataset can be found [here](https://ai.stanford.edu/~amaas/data/sentiment/), it needs preprocessing as done here [here](https://github.com/artemisart/bert-sentiment-IMDB).

## Recreating the experiments in the paper
### Adult dataset
Most results are in '01_Experiments_on_adult_dataset.ipynb' only the results for Table 2 are in '02_Table 2.ipynb'. The implementation of the 
actual explanations is in the `BII.py` file.

## IMDB dataset
We've precalculated the interactions of 2500 sentences in the dataset. They are saved in "max_sentences.pkl" and can be explored using the "04_IMDB_experiment_evaluation.ipynb" notebook. To recreate interactions the following steps need to be done:

1. Finetune BERT to the IMDB dataset (here we followed [this](https://github.com/artemisart/bert-sentiment-IMDB) Github page).
2. Create the input querries for BERT. This can be done with the code in the first half of "03_IMDB_experiments.ipynb".
3. Run BERT on these querries. This can be done with a variation of the script "script_for_BERT_predictions.sh"
4. Create the interactions from the predictions.  This can be done with the code in the second half of "03_IMDB_experiments.ipynb".