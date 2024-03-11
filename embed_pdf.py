from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from pymongo import MongoClient
from bson.binary import Binary
import pickle
import os
import json

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
    file = drive.CreateFile({'id': file_id})
    file.GetContentFile('temp.pdf')  # download file as 'temp.pdf'
    
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
    collection.insert_one({'file_name': file['title'], 'index': Binary(index_bytes)})

def embed_all_pdf_docs():
    drive = authenticate_with_drive()

    # Get the directory ID from the environment variable
    directory_id = os.getenv('DIRECTORY_ID')

    # Get the list of all files in the specified directory of Google Drive
    file_list = drive.ListFile({'q': f"'{directory_id}' in parents"}).GetList()

    # Filter the list to only include PDF files
    pdf_files = [file for file in file_list if file['title'].endswith('.pdf')]

    if pdf_files:
        for pdf_file in pdf_files:
            print(f"Embedding {pdf_file['title']}...")
            embed_document(file_id=pdf_file['id'], drive=drive)
            print("Done!")
    else:
        raise Exception("No PDF files found in the directory.")
    
import json
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

def get_all_index_files():
    drive = authenticate_with_drive()

    # Obtener el ID del directorio de las variables de entorno
    directory_id = os.getenv('DIRECTORY_ID')

    # Listar todos los archivos en el directorio
    file_list = drive.ListFile({'q': f"'{directory_id}' in parents and trashed=false"}).GetList()

    # Obtener los nombres de los archivos
    file_names = [file['title'] for file in file_list]

    return file_names