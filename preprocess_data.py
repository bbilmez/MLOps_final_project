"""
This python file reads the train/test/valiaton data and 
extracts predictor and target columns from these dataset.

Then, use DictVectorizer() to convert a dataframes into 
a numeric feature matrix suitable for machine learning algorithms.
"""

import os
import pickle
import argparse
import pandas as pd
from prefect import task, flow, get_run_logger
from sklearn.feature_extraction import DictVectorizer

@task(name="Read data")
def read_dataframe(filename: str):
    """
    Read csv data.
    """
    dataframe = pd.read_csv(filename)

    return dataframe

@task(name="Preprocess data")
def preprocess(dataframe: pd.DataFrame, dict_vect: DictVectorizer, fit_dv: bool = False):
    """
    Preprocess input data using dictionary vectorizer.
    """
    dicts = dataframe.to_dict(orient='records')
    if fit_dv:
        converted_dicts = dict_vect.fit_transform(dicts)
    else:
        converted_dicts = dict_vect.transform(dicts)
    return converted_dicts, dict_vect

@task(name="pickle")
def dump_pickle(obj, filename: str):
    """
    Pickle dictionary vectorizer.
    """
    with open(filename, "wb") as f_out:
        return pickle.dump(obj, f_out)

@flow(name="preprocess data")
def main(raw_data_path: str, dest_path: str):

    logger = get_run_logger()   

    # Load parquet files
    logger.info("Reading the data as dataframe")
    df_train = read_dataframe(
        os.path.join(raw_data_path, "train_data.csv")
    )
    df_test = read_dataframe(
        os.path.join(raw_data_path, "test_data.csv")
    )

    # Extract the target
    target = 'HeartDisease'
    y_train = df_train[target].values
    y_test = df_test[target].values

    del df_train[target]
    del df_test[target]

    # Fit the DictVectorizer and preprocess data
    logger.info("Vectorizing the dataframe")
    dict_vect = DictVectorizer()
    x_train, dict_vect = preprocess(df_train, dict_vect, fit_dv=True)
    x_test, _ = preprocess(df_test, dict_vect, fit_dv=False)

    # Create dest_path folder unless it already exists
    os.makedirs(dest_path, exist_ok=True)

    # Save DictVectorizer and datasets
    logger.info("Pickling dictvectorizer and datasets")
    dump_pickle(dict_vect, os.path.join(dest_path, "dv.pkl"))
    dump_pickle((x_train, y_train), os.path.join(dest_path, "train.pkl"))
    dump_pickle((x_test, y_test), os.path.join(dest_path, "test.pkl"))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_data_path", 
                        help="Location where the raw heart disease data was saved")
    parser.add_argument("--dest_path", help="Location where the resulting files will be saved")
    args = parser.parse_args()

    parameters = {
        "raw_data_path": args.raw_data_path,
        "dest_path": args.dest_path,
    }
    main(**parameters)
