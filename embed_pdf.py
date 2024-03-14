import os
import pickle
from bson.binary import Binary
from pymongo import MongoClient
from pymongo.errors import PyMongoError
import gridfs
from llm_helper import embed_and_store_document
from langchain_community.document_loaders.pdf import PagedPDFSplitter

def embed_document(file, filename):
    try:
        # Save file in MongoDB
        client = MongoClient(os.getenv('MONGODB_URI'))
        db = client[os.getenv('DB_NAME')]
        collection = db[os.getenv('MONGODB_COLLECTION_NAME')]
        fs = gridfs.GridFS(db)

        fs.put(file, filename=filename)
    except PyMongoError as e:
        print(f"Error while saving file to MongoDB: {e}")
        return

    # Determine the file type
    file_type = filename.split('.')[-1]

    if file_type == 'pdf':
        # Save the file to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp:
            temp.write(file)
            temp_file_path = temp.name
        # Pass the path of the temporary file to PagedPDFSplitter
        loader = PagedPDFSplitter(temp_file_path)
        source_pages = loader.load_and_split()
    elif file_type == 'txt':
        with open(filename, 'r') as f:  # Changed 'tempfile' to filename
            source_pages = [f.read()]
    else:
        print(f"Unsupported file type: {file_type}")
        return

    embed_and_store_document(source_pages, filename, collection)