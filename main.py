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
import json
# Hydrate the environment from the .env file
from dotenv import load_dotenv

# Importing flask app and processing libraries
from flask import Flask, render_template, request, session

# Initialize Google Cloud project and location
PROJECT_ID = os.environ['PROJECT_ID']
LOCATION = os.environ['LOCATION_ID']

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
def process_bills():
    try:
        filter_date = request.args.get("filter_date")

        # If a filter_date argument is passed, use it; otherwise, use 7 days ago from the current date.
        if filter_date:
            filtered_date = filter_date
        else:
            filtered_date = (datetime.now() - timedelta(days=8)).strftime("%Y-%m-%d")

        with pool.connect() as db_conn:
            bill_insert_stmt = sqlalchemy.text(
                f"SELECT summarized_bill, bills_inserted_date FROM {os.environ['DB_TABLE']} WHERE bills_inserted_date  >= '{filtered_date}'"
            )

            result = db_conn.execute(bill_insert_stmt)
            rows = result.fetchall()

        bills_data = [dict(eval(item[0]), bills_inserted_date=item[1]) for item in rows]
        columns = ['headline', 'story', 'twitter', 'bills_inserted_date']
        bills_data_df = pd.DataFrame(bills_data, columns=columns)


        # Replace '\n' with an empty string in the respective columns
        bills_data_df['headline'] = bills_data_df['headline'].str.replace('\n', '')
        bills_data_df['story'] = bills_data_df['story'].str.replace('\n', '')

        vector_lst = []
        def text_embedding() -> list:
            count = 0
            
            for text in range(len(bills_data_df['headline'])):
                """Text embedding with a Large Language Model."""
                if count > 59:
                    time.sleep(60)
                    count = 0
                    
                if not bills_data_df['headline'][text]:
                    embeddings = model.get_embeddings([bills_data_df['story'][text]])
                else:
                    embeddings = model.get_embeddings([bills_data_df['headline'][text]])
                    
                for embedding in embeddings:
                    vector = embedding.values
                    vector_lst.append(vector)
                    # print(f"Length of Embedding Vector: {len(vector)}")
                    count += 1
            return vector_lst
            
        bills_data_df['embedding'] = text_embedding()

        EMBED_DB_TABLE = os.environ['EMBED_DB_TABLE']
        
        bills_embed_insert_stmt = sqlalchemy.text(
                    f"INSERT INTO {EMBED_DB_TABLE} (headline, story, twitter, embedding, bills_inserted_date)"
                    "VALUES (:headline, :story, :twitter, :embedding, :bills_inserted_date)"
                )

        with pool.connect() as db_conn:
            for record in range(bills_data_df.shape[0]):
                # Insert the entry into the table
                db_conn.execute(bills_embed_insert_stmt, headline=bills_data_df['headline'][record], story=bills_data_df['story'][record], twitter=bills_data_df['twitter'][record], embedding=bills_data_df['embedding'][record], bills_inserted_date=bills_data_df['bills_inserted_date'][record])

        
        print(f" Total embeddings generated: {bills_data_df['embedding'].shape[0]} for records fetched from Cloud SQL table: {bills_data_df.shape}[0]")

       
        return "Data processing and upload completed successfully."

    except Exception as e:
        return f"An error occurred: {str(e)}"


if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
