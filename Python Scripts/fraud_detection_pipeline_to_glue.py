import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from awsglue.dynamicframe import DynamicFrame
from pyspark.sql.functions import col, when
from pyspark.ml.feature import VectorAssembler, StandardScaler, StringIndexer
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml import Pipeline
from pyspark.sql.types import DoubleType

## @params: [JOB_NAME]
args = getResolvedOptions(sys.argv, ['JOB_NAME'])
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args['JOB_NAME'], args)

# Modify function signature for Glue: Accept DynamicFrame, return DynamicFrameCollection
def MyTransform(dynamic_frame):
    try:
        # Convert DynamicFrameCollection to DataFrame for Glue
        df = dynamic_frame.toDF()

        # Convert all columns except 'Class' to DoubleType
        numeric_columns = [col for col in df.columns if col != 'Class']
        for column in numeric_columns:
            df = df.withColumn(column, col(column).cast(DoubleType()))

        # Normalize numeric features
        assembler = VectorAssembler(inputCols=numeric_columns, outputCol="numericFeatures")
        scaler = StandardScaler(inputCol="numericFeatures", outputCol="scaledFeatures", withStd=True, withMean=True)

        # Combine scaled numeric features
        finalAssembler = VectorAssembler(inputCols=["scaledFeatures"], outputCol="features")

        #Handle class imbalance by adjusting class weights
        class_counts = df.groupBy("Class").count().collect()
        total_count = sum([row['count'] for row in class_counts])
        weight_dict = {row['Class']: total_count / row['count'] for row in class_counts}
        
        # Add class weights to the DataFrame
        df = df.withColumn("weight", when(col("Class") == 0, weight_dict[0]).otherwise(weight_dict[1]))

        # Convert Class to numeric and create class weights
        indexer = StringIndexer(inputCol="Class", outputCol="label")
        
        # Define the RandomForestClassifier with class weights
        rf = RandomForestClassifier(labelCol="label", featuresCol="features", numTrees=100, 
                                    maxDepth=10, weightCol="weight")

        # Create a pipeline
        pipeline = Pipeline(stages=[assembler, scaler, finalAssembler, indexer, rf])

        # Fit the model
        model = pipeline.fit(df)

        # Modify model saving for Glue (use S3 instead of local path)
        model_path = "s3://myawsbucketpiplinefrauddetect/model/fraud_detection_model_latest/"
        model.write().overwrite().save(model_path)
        print(f'Model trained and saved to {model_path}')

        # Convert DataFrame back to DynamicFrame and return for Glue
        dynamic_frame = DynamicFrame.fromDF(df, glueContext, "dynamic_frame")
        
        return dynamic_frame    
    except Exception as e:
        print(f"Error: {e}")
        

# Glue job entry point

# Read the input CSV file from S3 into a DynamicFrame
dynamic_f = glueContext.create_dynamic_frame.from_options(
    connection_type="s3",
    connection_options={"paths": ["s3://myawsbucketpiplinefrauddetect/input/test_credit.csv"]},
    format="csv",
    format_options={"withHeader": True}
)

# Transform the data using MyTransform

MyTransform(dynamic_f)

# Write the transformed DynamicFrame back to S3 in Parquet format
glueContext.write_dynamic_frame.from_options(
    frame=dynamic_f,
    connection_type="s3",
    connection_options={"path": "s3://myawsbucketpiplinefrauddetect/output/"},
    format="parquet"
)

job.commit()