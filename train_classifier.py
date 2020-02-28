import sys
import nltk
import re
import pandas as pd
import numpy as np
import pickle
from typing import Tuple, List
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def load_data(database_filepath):
    '''
    input: database_filepath: Name of the sqlite database that the required table in saved in
    output: X: Messaged from the saved table
            Y: 36 Categories that a message can be classified as
            columns: Category names
    '''
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql("select * from ETLApp", engine)
    df.related.replace(2,1,inplace=True)
    X = df["message"]
    Y = df.iloc[:,4:]
    columns = Y.columns
    return X, Y, columns

def tokenize(text):
    '''
    input: text: a message
    output: tokens: tokenized version of the message
    
    Function tokenizes the text so that it can be used in a model
    '''
    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # tokenize text
    tokens = word_tokenize(text)
    # lemmatize andremove stop words
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return tokens


def build_model():
    '''
    Builds the model pipeline for classifting messages. 
    '''
    pipeline = Pipeline([
                    ("vect",CountVectorizer(tokenizer=tokenize)),
                    ('tfidf', TfidfTransformer()),
                    ('clf', MultiOutputClassifier(RandomForestClassifier()))
])
    parameters = {'clf__estimator__min_samples_leaf': [1,5],
              'clf__estimator__bootstrap': [True,False]
              }

    cv = GridSearchCV(pipeline, param_grid = parameters)

    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    input: model: The classification pipeline
           X_test: Messages test data
           Y_test: Classifications of X_test
           category_names: The categories that messages can belong to
     
     Function prints performance metrics of the model on the test data
    '''
    y_preds = pd.DataFrame(model.predict(X_test),columns = category_names)
    
    for i in range(Y_test.shape[1]):
        recall = recall_score(Y_test.iloc[:,i],y_preds.iloc[:,i])
        precision = precision_score(Y_test.iloc[:,i],y_preds.iloc[:,i])
        F1 = f1_score(Y_test.iloc[:,i],y_preds.iloc[:,i])
        print(" ")
        print(Y_test.columns[i])
        print("Recall:", recall)
        print("Precision", precision)
        print("F1 Score:", F1)
    


def save_model(model, model_filepath):
    
    '''
    input: model: classification pipeline
           model_filepath: location to save the model in
    Saves model as a pickel file
    '''
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    '''
    Runs all of the above functions to train the classifier
    '''
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()