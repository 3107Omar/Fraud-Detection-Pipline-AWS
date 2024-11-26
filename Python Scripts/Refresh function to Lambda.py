import json
import boto3
import os

sns_client = boto3.client('sns')
SNS_TOPIC_ARN = os.environ['SNS_TOPIC_ARN']
glue = boto3.client('glue')
gluejobname = "spark"

def lambda_handler(event, _context):
    
    print('event')
    
    bucket_name = event['Records'][0]['s3']['bucket']['name']
    object_key = event['Records'][0]['eventName']
    try:
        runId = glue.start_job_run(JobName=gluejobname)
        status = glue.get_job_run(JobName=gluejobname, RunId=runId['JobRunId'])
        print("Job Status : ", status['JobRun']['JobRunState'])
    except Exception as e:
        print(e)
        raise
    
    # Send an SNS notification
    response = sns_client.publish(
        TopicArn=SNS_TOPIC_ARN,
        Message=f'A new file has been uploaded to {bucket_name}: {object_key}',
        Subject='New File Uploaded'
    )
    
    return {
        'statusCode': 200,
        'body': f'Notification sent for file: {object_key}'
    }    





