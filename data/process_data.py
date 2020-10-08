"""
This script is an ETL pipeline for the processing of messages and its classifications

To run ETL pipeline that cleans data and stores in database:
python [path]process_data.py MESSAGES CATEGORIES DB,
for example:

python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db

It assumes the following:
* messages and categories are CSV files
* both have matching ids in variable called 'id'
* table in the database will be called 'classified_msgs'

"""

import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    loads messages and categories from csv file

    :param messages_filepath: full path of a csv file with messages
    :param categories_filepath: full path of a csv file with categories

    files need to have a common "id" variable

    :return: df

    (panda) Dataframe of the joined messages and categories source csvs
    """

    # load raw data
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    # merge on id
    df = pd.merge(messages, categories, on='id', how='inner')

    return df


def clean_data(df):
    """
    Cleans the dataframe returned by load_data.

    In particular it splits teh 'categories' field into multiple categories

    It assumes all rows have the exact same structure and categories

    :param df: dataframe returned by load_data

    :return: df: modified dataframe
    """

    # create a dataframe of the individual category columns
    categories = df['categories'].str.split(";", expand=True)

    # select the first row of the categories dataframe
    row = categories.iloc[1, :]

    # use this row to extract a list of new column names for categories
    category_colnames = row.apply(lambda x: x[:-2])
    categories.columns = category_colnames

    # convert category values to just numbers 0 or 1
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]

        # convert column from string to numeric
        categories[column] = categories[column].astype('int')

        # coerce anything greater than 1 into ==1
        categories.loc[categories[column] > 1, column] = 1

    # drop the original categories column from `df`
    df.drop('categories', axis=1, inplace=True)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)

    # drop duplicates
    df.drop_duplicates(inplace=True)

    return df


def save_data(df, database_filename):
    """
    Saves cleaned dataset into an SQLlite database table named 'classified_msgs')

    :param df: cleaned dataframe, returned by clean_data
    :param database_filename: full path of a db file

    :return: none
    """

    """
    from sqlalchemy docs:
    # sqlite://<nohostname>/<path>
    # where <path> is relative:
    engine = create_engine('sqlite:///foo.db')
    """
    # print('sqlite:///{}'.format(database_filename))
    engine = create_engine('sqlite:///{}'.format(database_filename))

    df.to_sql('classified_msgs', engine, index_label='id', index=False, if_exists='replace')


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories '
              'datasets as the first and second argument respectively, as '
              'well as the filepath of the database to save the cleaned data '
              'to as the third argument. \n\nExample: python process_data.py '
              'disaster_messages.csv disaster_categories.csv '
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
