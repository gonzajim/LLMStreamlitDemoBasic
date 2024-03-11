from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from pymongo import MongoClient
from bson.binary import Binary
import pickle
import os
import json
import boto3
from botocore.exceptions import NoCredentialsError

def authenticate_with_drive():
    # Cargar client_secrets del archivo JSON
    with open('./client_secrets.json', 'r') as f:
        client_secrets = json.load(f)

    # Autenticación con Google Drive
    gauth = GoogleAuth()
    gauth.client_config = client_secrets['web']  # Establecer la configuración del cliente
    gauth.CommandLineAuth()  # Genera URL para autenticación manual
    drive = GoogleDrive(gauth)

    return drive


def embed_document(file_id, drive):
    # Create a session using your AWS credentials
    s3 = boto3.client('s3')

    # The name of the bucket
    bucket_name = os.getenv('S3_BUCKET_NAME')

    try:
        s3.download_file(bucket_name, file_id, 'temp.pdf')
    except NoCredentialsError:
        print("No AWS credentials were found.")
        return

    loader = PagedPDFSplitter('temp.pdf')
    source_pages = loader.load_and_split()

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

    # Connect to MongoDB Atlas
    username = os.getenv('MONGODB_USERNAME')
    password = os.getenv('MONGODB_PASSWORD')

    client = MongoClient(f"mongodb+srv://{username}:{password}@cluster0.mongodb.net/myFirstDatabase?retryWrites=true&w=majority")
    db = client['mydatabase']
    collection = db['mycollection']

    # Convert the search index to bytes and store it in MongoDB
    index_bytes = pickle.dumps(search_index)
    collection.insert_one({'file_name': file_id, 'index': Binary(index_bytes)})


def embed_all_pdf_docs():
    # Create a session using your AWS credentials
    s3 = boto3.client('s3')

    # The name of the bucket
    bucket_name = os.getenv('S3_BUCKET_NAME')

    try:
        # List all the objects in the bucket
        objects = s3.list_objects(Bucket=bucket_name)['Contents']

        # Filter the list to only include PDF files
        pdf_files = [obj for obj in objects if obj['Key'].endswith('.pdf')]

        if pdf_files:
            for pdf_file in pdf_files:
                print(f"Embedding {pdf_file['Key']}...")
                embed_document(file_id=pdf_file['Key'], drive=s3)
                print("Done!")
        else:
            raise Exception("No PDF files found in the directory.")
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