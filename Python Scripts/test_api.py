import requests
import pytest
import pandas as pd
from sklearn.preprocessing import StandardScaler
import boto3
import csv
import io
from fastapi import FastAPI
# Create an S3 client
s3_client = boto3.client('s3', aws_access_key_id='', aws_secret_access_key='', aws_session_token='')

app = FastAPI()

# Read an object from the bucket
response = s3_client.get_object(Bucket='myawsbucketpiplinefrauddetect', Key='model/fraud_detection_model_latest/test_credit.csv')

# Read the object’s content as text
object_content = response['Body'].read().decode('utf-8')

# Process or use the content as needed
#print(object_content)

df = pd.read_csv(io.StringIO(object_content))

df.to_csv('local_output.csv', index=False)

# Define the base URL for the FastAPI application.
BASE_URL = "http://localhost:8000"

# Load the credit card fraud dataset
# Ensure the dataset path is correct and accessible
#dataset_path = "test_credit.csv"
df = pd.read_csv('local_output.csv')

# Select a sample row for testing
# Exclude the 'Class' column, which is the label
sample_row = df.drop(columns=['Class']).iloc[0]

# Standardize the numeric features using the same scaler settings as the model
scaler = StandardScaler()
scaler.fit(df.drop(columns=['Class']))  # Fit the scaler with the DataFrame, keeping feature names

# Convert the sample row to a DataFrame to retain feature names
sample_df = pd.DataFrame([sample_row])
print(f'Sample Row: {sample_row}')

# Transform the sample row using the fitted scaler
sample_scaled = scaler.transform(sample_df)

# Create a DataFrame with the expected column names
columns = df.drop(columns=['Class']).columns.tolist()  # Get the feature names
sample_scaled_df = pd.DataFrame(sample_scaled, columns=columns)

sample_scaled_list = sample_scaled_df.iloc[0].tolist()

# Define a fixture for sample input data.
@pytest.fixture
def sample_input():
    return {'features': sample_scaled_list}

# Test the /predict/ endpoint of the FastAPI application.
def test_predict(sample_input):
    print(f'Sample Input: {sample_input}')
    response = requests.post(f"{BASE_URL}/predict/", json=sample_input)
    assert response.status_code == 200
    assert "prediction" in response.json()
    prediction = response.json()["prediction"]
    assert isinstance(prediction, (int, float))  # Ensure the prediction is a number
