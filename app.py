from flask import Flask, request, jsonify
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
from flask_cors import CORS  # Import CORS from flask_cors
from sklearn.metrics import silhouette_score
from openai import OpenAI
import json
from sklearn.ensemble import IsolationForest
import csv
from io import StringIO
import re



app = Flask(__name__)
CORS(app, origins='*')

@app.route('/')
def index():
    return '<h1>Hello!</h1>'
    
@app.route('/bar')
def bar():
    return 'bar!'

def detect_outliers(data, features, contamination=0.1, random_state=None):
    """
    Detect outliers/anomalies in the dataset using Isolation Forest.

    Parameters:
        data (DataFrame): Input dataset.
        features (list): List of feature columns.
        contamination (float): The proportion of outliers in the dataset.
        random_state (int): Random seed for reproducibility.

    Returns:
        DataFrame: Dataset with an additional column 'outlier' indicating whether a data point is an outlier (1) or not (0).
    """
    # Copy the data to avoid modifying the original DataFrame
    data_copy = data.copy()
    
    # Select the feature columns
    X = data_copy[features]
    
    # Initialize the Isolation Forest model
    model = IsolationForest(contamination=contamination, random_state=random_state)
    
    # Fit the model
    model.fit(X)
    
    # Predict outliers
    data_copy['outlier'] = model.predict(X)
    
    return data_copy
  
@app.route('/outliers', methods=['POST'])
def outliers():

    data = request.json['data']
    
    df = pd.DataFrame(data)
    
    #Don't allow too big datasets in the beta version
    count_row = df.shape[0]
    if(count_row > 5000):
        return jsonify({
        'result': "not_OK",
        'error': 'too_many_rows', 
        'number_of_dataset_rows': count_row      
    })

    #DROP COLUMNS WITH CATEGORICAL VALUES
    for column in df.columns:
    # Check if the data type of the column is object (usually indicates string)
        if df[column].dtype == 'object':
        # Drop the column if it has string data type
            df.drop(column, axis=1, inplace=True)
            colsToUse = df.columns

    #FILL NaN WITH 0
    df.fillna(0, inplace=True)
    
    outliers = detect_outliers(df, colsToUse)
    
    outliers_json = outliers.to_json(orient='split')
    
    
   
    outliers_count = outliers['outlier'].value_counts().to_dict()
    # Access the count of the specific value you're interested in (1)
    outliers_count = outliers_count.get(-1, 0)
   
    # Return the cluster labels, cluster centers, and silhouette score as JSON
    return jsonify({
        #'column_names': colsToUse,
        'outliers': outliers['outlier'].tolist(),
        'outliers_count': outliers_count,
        'result': "OK",
        'number_of_dataset_rows': count_row    
    })
   
# Define a route for clustering
@app.route('/cluster', methods=['POST'])
def cluster():
    # Get input data and number of clusters from the request
    data = request.json['data']
    num_clusters = int(request.json['num_clusters'])

    # Convert data to a pandas DataFrame
    df = pd.DataFrame(data)
    
    #Don't allow too big datasets in the beta version
    count_row = df.shape[0]
    if(count_row > 5000):
        return jsonify({
        'result': "not_OK",
        'error': 'too_many_rows', 
        'number_of_dataset_rows': count_row      
    })
    
    #Don't allow too many clusters
    if(num_clusters > 10):
        return jsonify({
        'result': "not_OK",
        'error': 'too_many_selected_clusters', 
        'number_of_dataset_rows': count_row      
    })
    

    # One-hot encode categorical variables
    #df = pd.get_dummies(df)

    # FILL NaN WITH 0
    df.fillna(0, inplace=True)

    # Perform K-means clustering
    kmeans = KMeans(n_clusters=num_clusters)
    clusters = kmeans.fit_predict(df)

    # Get cluster centers
    cluster_centers = kmeans.cluster_centers_

    # Calculate silhouette score
    silhouette_avg = silhouette_score(df, clusters)

    # Return the cluster labels, cluster centers, and silhouette score as JSON
    return jsonify({
        'cluster_centers': cluster_centers.tolist(),
        'silhouette_score': silhouette_avg,
        'clusters': clusters.tolist(),
        'result': "OK",
        'number_of_dataset_rows': count_row    
    })
   
@app.route('/plotdata', methods=['POST'])
def plotdata():

    data = request.json['data']
    

    df=''
        
    def read_csv_from_string(csv_string):
        # Convert the CSV-like string into a file-like object
        csv_file = StringIO(csv_string)
        # Read the CSV data into a DataFrame
        df = pd.read_csv(csv_file)
        return df

    df = read_csv_from_string(data)
    print(df)

    
    for column in df.columns:
        if df[column].dtype == 'object':
            df[column] = df[column].str.replace(',', '')
        print(f"Column '{column}' has data type: {df[column].dtype}")
    
    for column in df.columns:
        # Attempt to convert the column to numeric
        try:
            df[column] = pd.to_numeric(df[column])
        except ValueError:
            # If conversion fails, it means the column should not be numeric
            pass

    # Display the DataFrame info to confirm the data types
    print("+++++++++++++")
    print(df.info())
    
    threshold = 0.3  # Adjust as needed

    # Iterate over each column in the DataFrame
    for column in df.columns:
        # Check if the data type of the column is object (usually indicates string)
        if df[column].dtype == 'object':
            # Calculate the proportion of categorical values in the column
            categorical_proportion = df[column].apply(lambda x: isinstance(x, str)).mean()
            # Drop the column if the proportion exceeds the threshold
            if categorical_proportion > threshold:
                df.drop(column, axis=1, inplace=True)

    # Update colsToUse after dropping columns
    colsToUse = df.columns
    
    #print("===============")
    #print(colsToUse)
    #FILL NaN WITH 0
    df.fillna(0, inplace=True)
    
    
    plot_ready_data = df.to_json()
    #print("===================================")
    #print(plot_ready_data)
    

    # Return the cluster labels, cluster centers, and silhouette score as JSON
    return jsonify({
        'plot_ready_data': plot_ready_data 
    })
    
@app.route('/explaincluster', methods=['POST'])
def explaincluster():
    API_KEY = open("API_KEY.txt", "r").read().strip()
    #openai.api_key = API_KEY
    
    client = OpenAI(api_key=API_KEY)

    data = request.json['data']

    completion = client.chat.completions.create(
      model="gpt-3.5-turbo",
      messages=[
        {"role": "system", "content": "You are a data scientist."},
        {"role": "user", "content": "These are the centers of my groups that I identified through clustering: "+ data+ ". Describe each group without plagiarism. Give to each group a descriptive name. Mark each group's description end with \n"}
      ],
      max_tokens=500 
    )
    gpt_result =  completion.choices[0].message.content


    return jsonify({
        'result': "OK",
        'gpt_result': gpt_result
    })

@app.route('/explaincluster', methods=['OPTIONS'])
def handle_options():
    # Add CORS headers to response
    response = jsonify({})
    response.headers.add('Access-Control-Allow-Origin', 'http://mlbro.scienceontheweb.net')
    response.headers.add('Access-Control-Allow-Methods', 'POST')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
    return response
    
    
@app.route('/suggestactions', methods=['POST'])
def suggestactions():
    API_KEY = open("API_KEY.txt", "r").read().strip()
    #openai.api_key = API_KEY
    
    client = OpenAI(api_key=API_KEY)

    goal = request.json['goal']
    context = request.json['context']
    cluster = request.json['cluster']
    
    print(goal)

    completion = client.chat.completions.create(
      model="gpt-3.5-turbo",
      messages=[
        {"role": "system", "content": context},
        {"role": "user", "content": "These are the centers of my groups that I identified through clustering: "+ cluster+ ". Suggest me 5 actions so that I can achieve my goal: "+ goal + ". Mark each action's description end with \n"}
      ],
      max_tokens=1000 
    )
    gpt_result =  completion.choices[0].message.content


    return jsonify({
        'result': "OK",
        'gpt_result': gpt_result
    })

@app.route('/suggestactions', methods=['OPTIONS'])
def handle_options_suggest_actions():
    # Add CORS headers to response
    response = jsonify({})
    response.headers.add('Access-Control-Allow-Origin', 'http://mlbro.scienceontheweb.net')
    response.headers.add('Access-Control-Allow-Methods', 'POST')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
    return response

@app.route('/explainoutliers', methods=['POST'])
def explainoutliers():
    API_KEY = open("API_KEY.txt", "r").read().strip()
    #openai.api_key = API_KEY
    
    client = OpenAI(api_key=API_KEY)

    data = request.json['data']

    df = pd.read_csv(data)
    #print(df.head(1))

    completion = client.chat.completions.create(
      model="gpt-3.5-turbo",
      messages=[
        {"role": "system", "content": "You are a data scientist."},
        {"role": "user", "content": "These are the datasets outliers: "+ data+ ". Give me insights on the outlier rows, maybe point out significant things and do a trend analysis.Mark each sentence end with \n"}
      ],
      max_tokens=500 
    )
    gpt_result =  completion.choices[0].message.content


    return jsonify({
        'result': "OK",
        'gpt_result': gpt_result
    })

@app.route('/explainoutliers', methods=['OPTIONS'])
def handle_options_outliers():
    # Add CORS headers to response
    response = jsonify({})
    response.headers.add('Access-Control-Allow-Origin', 'http://mlbro.scienceontheweb.net')
    response.headers.add('Access-Control-Allow-Methods', 'POST')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
    return response

@app.route('/exportsql', methods=['POST'])
def exportsql():
    API_KEY = open("API_KEY.txt", "r").read().strip()
    #openai.api_key = API_KEY
    
    client = OpenAI(api_key=API_KEY)

    data = request.json['data']

    completion = client.chat.completions.create(
      model="gpt-3.5-turbo",
      messages=[
        {"role": "system", "content": "You are a data scientist. You deliver SQL code only."},
        {"role": "user", "content": "These are the centers of my clusters: "+ data+ ". create for me the SQL code that will allocate a new item to the appropriate cluster. Provide only the code and avoid other messages. In the code put <br> to mark the change of line."}
      ],
      max_tokens=500 
    )
    gpt_result =  completion.choices[0].message.content


    return jsonify({
        'result': "OK",
        'gpt_result': gpt_result
    })

@app.route('/exportsql', methods=['OPTIONS'])
def handle_options_sql():
    # Add CORS headers to response
    response = jsonify({})
    response.headers.add('Access-Control-Allow-Origin', 'http://mlbro.scienceontheweb.net')
    response.headers.add('Access-Control-Allow-Methods', 'POST')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
    return response
    
@app.route('/plotdata', methods=['OPTIONS'])
def handle_options_plotdata():
    # Add CORS headers to response
    response = jsonify({})
    response.headers.add('Access-Control-Allow-Origin', 'http://mlbro.scienceontheweb.net')
    response.headers.add('Access-Control-Allow-Methods', 'POST')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
    return response
    
# Run the app
if __name__ == '__main__':
    app.run(debug=True)