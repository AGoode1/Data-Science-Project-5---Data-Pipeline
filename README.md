# Data-Science-Project-5---Data-Pipeline
Udacity Data Scientist nano-degree: Data Pipeline Project


The project takes in messages from disaster events, cleans the data and build a modelling pipeline model to categorise new messages. This is then presented as an online app. 

process_data.py imports the messages data, cleans the data and saves the data in an sqlite database.

train_classifier.py uploads the dataset from the sqlite database and builds a modelling pipeline with it, saving the model as a pickle file. 

run.py creates the web app. 

To run the project:

1. Run the following commands in to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        'python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db'
    - To run ML pipeline that trains classifier and saves
        'train_classifier.py DisasterResponse.db classifier.pkl'

2. Run the following command in the app's directory to run your web app.
    'run.py'

3. Go to http://0.0.0.0:3001/

If using the udacity terminal:

Once the app is running (python run.py):
a) Open another terminal and type env|grep WORK this will give you the spaceid it will start with view*** and some characters after that
b)Now open your browser window and type https://viewa7a4999b-3001.udacity-student-workspaces.com, replace the whole viewa7a4999b with your space id. 
c)Press enter and the app will run


If using a Local Machine

Once the app is running (python run.py) go to to localhost:3001 and app will run
