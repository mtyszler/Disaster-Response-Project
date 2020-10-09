"""
This script is an primarily a ML pipeline for the processing training of a model to classify 
msgs in multi-categories, and saves it in a pickle file

To run ML pipeline that loads data prepared by the ETL (see process_data.py)
python [path]train_classifier.py DB PKL,
for example:

python models/train_classifier.py data/DisasterResponse.db models/best_model.pkl

It assumes the following:
* DB has been prepared by the ETL
* table in the database will be called 'classified_msgs'
* 'classified_msgs' has messages (in a field callsed 'messages') and classifications (starting at iloc 4)


"""

import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
import re
import pickle

from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer

from sklearn.pipeline import Pipeline

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import make_scorer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.multioutput import MultiOutputClassifier

from sklearn.model_selection import GridSearchCV

import nltk

# download supporting files
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


def average_accuracy(Y_real, Y_pred):
    """
    Computes average accuracy across all categories:
    
    inputs:
    Y_real: correct classifications
    Y_pred: predicted classifications
    
    Return:
    accuracy
    """
    
    return (Y_real == Y_pred).sum().sum()/Y_real.size

def average_f1(Y_real, Y_pred, avg = 'macro'):
    """
    Computes average f1-score across all categories:
    
    inputs:
    Y_real: correct classifications
    Y_pred: predicted classifications
    avg: Default == 'macro'. See f1_score doc for other options
    
    Return:
    accuracy
    """
    
    f1 = []
    
    # iterate
    for i, category in enumerate(Y_real):
        f1.append(f1_score(Y_test[category], Y_pred[:, i], average = avg))
        
    
    return np.mean(f1)

def naive_predictions(Y_test):
	"""
	Creates naive predictions, by predicting the majority class

	inputs:
	Y_test: (correct) targets

	return
	naive_pred: naive predictions

	"""
	
	naive_pred = Y_test.copy()
	for category in Y_test.columns:
    
    # check accucary of predicting TRUE to all
    categ_acc = Y_test[category].mean()
    
    # predict the majority class:
    if categ_acc>0.5:
        naive_pred[category] = 1
    else:
        naive_pred[category] = 0

    return naive_pred

def print_classification_report(Y_real, Y_pred):
    """
    Prints classification report per catetory:
    
    inputs:
    Y_real: correct classifications
    Y_pred: predicted classifications
    
    Return:
    none
    """
    # create lists:
    acc = []
    f1 = []
    
    # iterate
    for i, category in enumerate(Y_real):
        print("Category: ",category)
        print(classification_report(Y_test[category], Y_pred[:, i]))
        f1.append(f1_score(Y_test[category], Y_pred[:, i], average = 'macro'))
        acc.append(accuracy_score(Y_test[category], Y_pred[:, i]))

    print("-------------------------------------------------------")    
    print("Mean accuracy score: {:.4f}".format(np.mean(acc)))
    print("Mean f1-score (macro): {:.4f}".format(np.mean(f1)))


def load_data(database_filepath):
	"""
    loads messages and categories from database

    :param database_filepath: full path of a db file with messages and categories

    :return: 
    pandas DataFrames:
    X: messages
    Y: targets

    """
    
    # print('sqlite:///{}'.format(database_filepath))
    engine = create_engine('sqlite:///{}'.format(database_filepath))
	
	df = pd.read_sql('SELECT * FROM classified_msgs', engine)

	X = df['message']
	Y = df.iloc[:,4:]

	return X, Y
	

def tokenize(text):
	"""
	Converts a text to tokens, by the following pipeline:

	* Normalize case and remove punctuations
	* split into words
	* remove stop words (english)
	* lemmatize
	* stems


	input:
	text: a string

	returs:
	tokenize string, in a list

	"""

    # prep nltk transformation objects
    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()
    
    # normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # tokenize text
    tokens = word_tokenize(text)
    
    # lemmatize and remove stop words
    lemmed = [lemmatizer.lemmatize(word, pos='v') for word in tokens if word not in stop_words]
    
    # Reduce words to their stems
    stemmed = [stemmer.stem(word) for word in lemmed]

    return stemmed


def build_model():
	"""
	Creates a ML pipeline with GridSearchCV

	input: nothing

	returns:
	GridSearchCV object


	"""

    pipeline = Pipeline([
    	('vect' , CountVectorizer(tokenizer=tokenize)),
        ('tfidf' , TfidfTransformer()),
        ('clf' , MultiOutputClassifier(AdaBoostClassifier(random_state = 42)))
	], verbose = True)

	# parameters to tune
	parameters = {
	    'tfidf__use_idf':[True, False],
	    'clf__estimator__n_estimators':[10, 50, 100, 150, 200, 500],
	    'clf__estimator__learning_rate':[1.0,1.5,2.0]
	}

	# scorer using the average_f1
	scorer = make_scorer(average_f1)

	# Grid Search object
	cv = GridSearchCV(pipeline, param_grid=parameters, scoring = scorer, verbose = 2, n_jobs = -1)

	return cv


def evaluate_model(model, X_test, Y_test):
	"""
	shows perfomance metrics of the (best) model, compared to a naive predictor

	inputs:
	model: already fitted
	X_test: test features
	Y_test: test targets

	returns: 
	none

	"""

	# create naive predictions
	naive_pred = naive_predictions(Y_test)
    
    # create model predictions
    Y_pred = model.predict(X_test)

    print("Naive accuracy: ", average_accuracy(Y_test, naive_pred))
	print("Optmized model accuracy: ", average_accuracy(Y_test, Y_pred))

	print("Naive f1-score: ", average_f1(Y_test, naive_pred.to_numpy()))
	print("Optmized model f1-score: ", average_f1(Y_test, Y_pred))

	print("Detailed report:")
    print_classification_report(Y_test, Y_pred)

def save_model(model, model_filepath):
	"""
	Saves model to a pickle file

	inputs:
	model: already fitted
	model_filepath: full path for the pickle file

	returns: 
	none
	"""
    
	pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)

        print("Best model:")
		print(model.best_estimator_)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test)

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