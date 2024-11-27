from fastapi import FastAPI
from pydantic import BaseModel
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.ml.linalg import Vectors
import os
from pyspark.ml import PipelineModel
from pyspark.sql import SparkSession
import boto3


class InputData(BaseModel):
    features: list[float]

app = FastAPI()

# Initialize Spark session
spark = SparkSession.builder.appName("FraudDetectionAPI").getOrCreate()

model = None
model_path = 'model/fraud_detection_model_latest/'  # Temporary local path for model

@app.on_event('startup')
async def load_model():
    """
    Load the model from cloud storage into the application at startup.
    """
    global model

    # Cloud storage details (S3 bucket and key)
    s3_bucket = os.getenv('S3_BUCKET', 'myawsbucketpiplinefrauddetect')
    s3_key = os.getenv('S3_KEY', 'model/fraud_detection_model_latest')

    # Ensure the local directory exists
    os.makedirs(model_path, exist_ok=True)

    # Download model files from S3 to local path
    s3 = boto3.client('s3')
    for obj in s3.list_objects_v2(Bucket=s3_bucket, Prefix=s3_key).get('Contents', []):
        file_key = obj['Key']
        relative_path = file_key[len(s3_key):]
        local_file_path = os.path.join(model_path, relative_path)

    # Load the model
    model = PipelineModel.load(model_path)
    print(f"Model loaded successfully from S3 bucket {s3_bucket}!")

@app.post('/predict/')
def predict(data: InputData):
    """
    Perform prediction using the loaded model.
    """
    if model is None:
        return {'error': 'Model is not loaded. Please check the model path or train the model.'}

    # Define feature columns
    columns = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11',
               'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22',
               'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']

    # Create a Spark DataFrame for prediction
    df = spark.createDataFrame([data.features], columns)

    # Make prediction
    predictions = model.transform(df)
    prediction = predictions.select("prediction").collect()[0][0]
    return {'prediction': prediction}

