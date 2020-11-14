# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import pandas as pd
from util_dataset import to_interim, to_processed

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('interim_filepath', type=click.Path())
@click.argument('processed_filepath', type=click.Path())
def main(input_filepath, interim_filepath, processed_filepath, ):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    ### RAW DATA PULL ###
    train_data = pd.read_csv(input_filepath + '/train.csv', index_col='id')
    submission_data = pd.read_csv(input_filepath + '/test.csv', index_col='id')

    ### PROCESSING ###
    to_interim(train_data, submission_data, interim_filepath)

    ### SAVED TO PROCESSED FILEPATH ###
    to_processed(train_data, submission_data, processed_filepath)


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
