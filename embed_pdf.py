from pymongo import MongoClient
from bson.binary import Binary
import pickle
import os
import json
import boto3
from botocore.exceptions import NoCredentialsError

def embed_document(file_id, drive, s3, bucket_name, collection):
    try:
        s3.download_file(bucket_name, file_id, 'tempfile')
    except NoCredentialsError:
        print("No AWS credentials were found.")
        return

    # Determine the file type
    file_type = file_id.split('.')[-1]

    if file_type == 'pdf':
        loader = PagedPDFSplitter('tempfile')
        source_pages = loader.load_and_split()
    elif file_type == 'txt':
        with open('tempfile', 'r') as f:
            source_pages = [f.read()]
    else:
        print(f"Unsupported file type: {file_type}")
        return

    embedding_func = OpenAIEmbeddings()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        length_function=len,
        is_separator_regex=False,
        separators=["\n\n", "\n", " ", ""],
    )
    source_chunks = text_splitter.split_documents(source_pages)
    search_index = FAISS.from_documents(source_chunks, embedding_func)

    # Convert the search index to bytes and store it in MongoDB
    index_bytes = pickle.dumps(search_index)
    collection.insert_one({'file_name': file_id, 'index': Binary(index_bytes)})

def embed_all_docs():
    # Create a session using your AWS credentials
    s3 = boto3.client('s3')

    # The name of the bucket
    bucket_name = os.getenv('S3_BUCKET_NAME')

    # Create a MongoDB client
    mongo_client = MongoClient(os.getenv('MONGODB_URI'))
    db = mongo_client[os.getenv('MONGODB_DB_NAME')]
    collection = db[os.getenv('MONGODB_COLLECTION_NAME')]

    try:
        # List all the objects in the bucket
        objects = s3.list_objects(Bucket=bucket_name)['Contents']

        if objects:
            for obj in objects:
                print(f"Embedding {obj['Key']}...")
                embed_document(file_id=obj['Key'], drive=s3, s3=s3, bucket_name=bucket_name, collection=collection)
                print("Done!")
        else:
            raise Exception("No files found in the directory.")
    except NoCredentialsError:
        print("No AWS credentials were found.")

def get_all_index_files():
    # Create a session using your AWS credentials
    s3 = boto3.client('s3')

    # The name of the bucket
    bucket_name = os.getenv('S3_BUCKET_NAME')

    try:
        # List all the objects in the bucket
        objects = s3.list_objects(Bucket=bucket_name)['Contents']

        # Get the names of the files
        file_names = [obj['Key'] for obj in objects]

        return file_names
    except NoCredentialsError:
        print("No AWS credentials were found.")
        return []  # Return an empty list if no credentials are found
    except Exception as e:
        print(f"An error occurred: {e}")
        return []  # Return an empty list if an error occurs