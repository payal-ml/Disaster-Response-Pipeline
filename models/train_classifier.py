import sys
import pandas as pd
from sqlalchemy import create_engine

from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report
import pickle
import nltk
import re
from nltk.stem import WordNetLemmatizer
nltk.download("stopwords")
nltk.download('wordnet')
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV


def load_data(database_filepath):
    """
    load data from database
    """
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('messages', engine)  
    X = df["message"]    
    category_names=['related', 'request', 'offer','aid_related', 'medical_help', 'medical_products', 'search_and_rescue',
           'security', 'military', 'child_alone', 'water', 'food', 'shelter','clothing', 'money', 'missing_people',
            'refugees', 'death', 'other_aid','infrastructure_related', 'transport', 'buildings', 'electricity',
           'tools', 'hospitals', 'shops', 'aid_centers', 'other_infrastructure','weather_related', 'floods', 'storm', 
            'fire', 'earthquake', 'cold','other_weather', 'direct_report']
    Y = df[category_names]
    X=tokenize(X)
    return X,Y,category_names

def clean_text(text):
    '''
    cleaning the text
    '''
    text = text.lower()
    text = text.strip()    
    text = re.sub(r"[-()\"#/$%&*<@>;:{}`+=~|.!?,'0-9]", "", text)
    return text

def tokenize(text):
    '''
    Removing stopword from text and applying lemmatization
    '''
    text = text.apply(lambda x: clean_text(x))
    stop = set(nltk.corpus.stopwords.words('english'))
    # Create WordNetLemmatizer object
    wnl = WordNetLemmatizer()
    text = text.apply(lambda x: " ".join(x for x in x.split() if x not in stop))
    text = text.apply(lambda x: " ".join([wnl.lemmatize(word) for word in x.split()]))   #lemmatization
    return text 


def build_model():
    '''
    build model pipeline and grid serach for optimal parameter
    '''
    pipeline = Pipeline([('tfidf',TfidfVectorizer()),
                     ('clf',MultiOutputClassifier(RandomForestClassifier(random_state=1)))
    
                ])
    parameters ={
        
        
        'clf__estimator__min_samples_split': [2, 3, 4],
        'clf__estimator__criterion' :['gini', 'entropy'],
        
    }

    cv = GridSearchCV(pipeline, param_grid=parameters, n_jobs=-1)

    return cv
def evaluate_model(model, X_test, Y_test, category_names):
    '''
    predicting on test data and adding f1 scroe, recall and precision for classes
    '''
    # predict on test data
    Y_pred = model.predict(X_test)
    for i in range(len(category_names)):
        print(f'{category_names[i]} classification_report:')
        print(classification_report(Y_test.iloc[:,i], Y_pred[:,i]))


def save_model(model, model_filepath):
    '''
    Export model as a pickle file

    '''
   
    pickle.dump(model,open(model_filepath,'wb'))


def main():
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