import os
import pickle
from bson.binary import Binary
from pymongo import MongoClient
from pymongo.errors import PyMongoError
import gridfs
from langchain_community.document_loaders import PyPDFLoader as PagedPDFSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
import tempfile

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

    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=OPENAI_API_KEY)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        length_function=len,
        is_separator_regex=False,
        separators=["\n\n", "\n", " ", ""],
    )
    source_chunks = text_splitter.split_documents(source_pages)
    search_index = FAISS.from_documents(source_chunks, embeddings)

    try:
        # Store it in MongoDB
        collection.insert_one({'faiss_index': search_index.serialize_to_bytes()})
    except PyMongoError as e:
        print(f"Error inserting into MongoDB: {e}")

    return search_index

def load_embeddings_and_index():
    try:
        # Connect to MongoDB
        client = MongoClient(os.getenv('MONGODB_URI'))
        db = client[os.getenv('DB_NAME')]
        collection = db[os.getenv('MONGODB_COLLECTION_NAME')]

        # Retrieve the document
        doc = collection.find_one()

        # Load the FAISS index from the document
        faiss_index_bytes = doc['faiss_index']
        search_index = FAISS.deserialize_from_bytes(faiss_index_bytes)

        return search_index
    except PyMongoError as e:
        print(f"Error retrieving from MongoDB: {e}")
        return None