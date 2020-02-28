import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    Input: messages_filepath: a csv file containing messages
           categories_filepath: a csv file containign categories for the messages
         
    Output: df: Merged Pandas dataframe
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, on = "id", how = "inner")
    return df


def clean_data(df):
    '''
    Input: df: dataframe containing messages and associated categories
    Output: df: Cleaned data file.
    
    Takes the data frame, replaces column names with disaster categories, and encodes the data as binary
    '''
    categories = df["categories"].str.split(pat = ";", expand = True)
    
    row = pd.Series(categories.iloc[0,:])
    category_colnames = row.apply(lambda x: x[0:-2])
    categories.columns = category_colnames

    for column in categories:
        categories[column] = categories[column].astype(str).str[-1]
        categories[column] = categories[column].astype(int)

    df = df.drop(["categories"], axis = 1)
    df = pd.concat([df,categories], axis = 1)
    df = df.drop_duplicates()
    return df
    
def save_data(df, database_filename):
    '''
    input: df: Cleaned dataframe
           database_filename: Database name that table will be saved in
    output: None
    Function will save dataframe df as ETLApp in the chosen sqlite database. 
    '''
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('ETLApp', engine, if_exists = 'replace', index=False)  


def main():
    '''
    Calls all above functions
    '''
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
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()