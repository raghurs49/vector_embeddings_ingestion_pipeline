from google.cloud.sql.connector import Connector
from google.cloud import storage
import sqlalchemy
from sqlalchemy import text
import pandas as pd
from datetime import datetime, timedelta
import vertexai
from vertexai.preview.language_models import TextEmbeddingModel
import time
import numpy as np
import os
# Hydrate the environment from the .env file
from dotenv import load_dotenv

# Importing flask app and processing libraries
from flask import Flask, render_template, request, session

# Initialize Google Cloud project and location
PROJECT_ID = os.environ['PROJECT_ID']
LOCATION = os.environ['LOCATION_ID']
BUCKET_NAME = os.environ['BUCKET_NAME']

# Initialize Vertex AI
vertexai.init(project=PROJECT_ID, location=LOCATION)
model = TextEmbeddingModel.from_pretrained("textembedding-gecko@001")

load_dotenv()

app = Flask(__name__)
app.secret_key = 'flask_secret_key'

# Initialize Cloud SQL Connector
connector = Connector()

# Connect to Cloud SQL database
def connect_to_db():
    conn = connector.connect(
        os.environ["DB_INSTANCE"],
        "pg8000",
        user=os.environ['DB_USER'],
        password=os.environ['DB_PASSWORD'],
        db=os.environ['DB_NAME']
    )
    return conn

# Create a connection pool for Cloud SQL
pool = sqlalchemy.create_engine(
    "postgresql+pg8000://",
    creator=connect_to_db,
)

@app.route('/embedd', methods=['GET'])
def process_bills(request):
    try:
        filter_date = request.args.get("filter_date")

        # If a filter_date argument is passed, use it; otherwise, use 7 days ago from the current date.
        if filter_date:
            filtered_date = filter_date
        else:
            filtered_date = (datetime.now() - timedelta(days=8)).strftime("%Y-%m-%d")

        with pool.connect() as db_conn:
            bill_insert_stmt = sqlalchemy.text(
                f"SELECT * FROM bills WHERE {os.environ['DB_TABLE']} > '{filtered_date}'"
            )

            result = db_conn.execute(bill_insert_stmt)
            rows = result.fetchall()

        columns = ['headline', 'title', 'twitter', 'bills_inserted_date']
        bills_data_df = pd.DataFrame(rows, columns=columns)

        # Replace '\n' with an empty string in the respective columns
        bills_data_df['headline'] = bills_data_df['headline'].str.replace('\n', '')
        bills_data_df['title'] = bills_data_df['title'].str.replace('\n', '')

        def get_embedding(text):
            get_embedding.counter += 1
            try:
                if get_embedding.counter % 1 == 0:
                    time.sleep(5)
                return model.get_embeddings([text])[0].values.tolist()
            except:
                return []

        get_embedding.counter = 0

        bills_data_df["embedding"] = bills_data_df["headline"].apply(lambda x: get_embedding(x))
        
        # Define the path to the local file you want to upload
        local_file_path = f"embeddings_bills_{datetime.now().strftime('%Y-%m-%d')}.csv"

        # Save DataFrame to a CSV file
        bills_data_df["embedding"].to_csv(local_file_path, index=False)

        # Upload data to GCS
        upload_data_to_gcs(local_file_path)

        return "Data processing and upload completed successfully."

    except Exception as e:
        return f"An error occurred: {str(e)}"

def upload_data_to_gcs(local_file_path):
    try:
        # Create a client to interact with GCS
        storage_client = storage.Client(project=PROJECT_ID)
         # Define the path to the local file you want to upload

        # Define the name you want to give the file in GCS
        blob_name = f"embeddings-folder/embeddings_bills_{datetime.now().strftime('%Y-%m-%d')}.csv"

        # Get a reference to the bucket
        bucket = storage_client.bucket(bucket_name=BUCKET_NAME)

        # Upload the local file to GCS
        blob = bucket.blob(blob_name)
        blob.upload_from_filename(local_file_path)

        return f'File {local_file_path} uploaded to GCS bucket {BUCKET_NAME} as {blob_name}'
    except Exception as e:
        return f"Error uploading to GCS: {str(e)}"


if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))