import os
import pickle
from bson.binary import Binary
from pymongo import MongoClient
from pymongo.errors import PyMongoError
import gridfs
from langchain_community.document_loaders.pdf import PagedPDFSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_faiss import FAISS  # Assuming FAISS is in this module

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
        loader = PagedPDFSplitter(file)
        source_pages = loader.load_and_split()
    elif file_type == 'txt':
        with open(filename, 'r') as f:  # Changed 'tempfile' to filename
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

    try:
        # Convert the search index to bytes
        index_bytes = pickle.dumps(search_index)
    except (pickle.PicklingError, AttributeError) as e:
        print(f"Error serializing search index: {e}")
        return

    try:
        # Store it in MongoDB
        collection.insert_one({'file_name': filename, 'index': Binary(index_bytes)})
    except PyMongoError as e:
        print(f"Error inserting into MongoDB: {e}")