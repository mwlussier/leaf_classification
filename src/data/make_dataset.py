# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import pandas as pd
from util_dataset import to_interim, to_processed
from preprocess_dataset import complete_preprocessing

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('interim_filepath', type=click.Path())
@click.argument('processed_filepath', type=click.Path())
def main(input_filepath, interim_filepath, processed_filepath):
    """
        Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    ### RAW DATA PULL ###
    train_data = pd.read_csv(input_filepath + '/train.csv', index_col='id')
    submission_data = pd.read_csv(input_filepath + '/test.csv', index_col='id')

    ### PROCESSING ###
    to_interim(train_data, seperate_label=True, file_name='/train.csv')
    to_interim(submission_data, file_name='/submission.csv')

    train_simple, submission_simple = complete_preprocessing(train_data, submission_data)
    train_pca_50, submission_pca_50 = complete_preprocessing(train_data, submission_data, pca=50)
    train_pca_100, submission_pca_100 = complete_preprocessing(train_data, submission_data, pca=100)
    train_pca_150, submission_pca_150 = complete_preprocessing(train_data, submission_data, pca=150)
    train_fselection, submission_fselection = complete_preprocessing(train_data, submission_data,
                                                                     features_selection=True)

    ### SAVED TO PROCESSED FILEPATH ###
    # preprocessed Simple
    to_processed(train_simple, submission_simple,
                 train_suffixe='train_simple', submission_suffixe='submission_simple')
    # preprocessed PCA(50)
    to_processed(train_pca_50, submission_pca_50,
                 train_suffixe='train_pca_50', submission_suffixe='submission_pca_50')
    # preprocessed PCA(100)
    to_processed(train_pca_100, submission_pca_100,
                 train_suffixe='train_pca_100', submission_suffixe='submission_pca_100')
    # preprocessed PCA(150)
    to_processed(train_pca_150, submission_pca_150,
                 train_suffixe='train_pca_150', submission_suffixe='submission_pca_150')
    # preprocessed Features Selection
    to_processed(train_fselection, submission_fselection,
                 train_suffixe='train_feature_selection', submission_suffixe='submission_feature_selection')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]
    print(project_dir)
    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())
    main()
