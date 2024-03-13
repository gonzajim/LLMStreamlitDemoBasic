import streamlit as st
import os
import embed_pdf

# get openai api key from environment variable
openapi_key = os.getenv("OPENAPI_KEY")

# Variable global para almacenar el tamaño de los chunks
chunk_size = st.sidebar.text_input("Chunks:", value="500")

# Convertir el tamaño de los chunks a un entero
try:
    chunk_size = int(chunk_size)
except ValueError:
    st.sidebar.error("El tamaño de los chunks debe ser un número entero.")
    chunk_size = 500  # Valor por defecto

# Obtener la lista de ficheros de S3
try:
    s3_files = embed_pdf.get_all_index_files()
    s3_files_str = "\n".join(str(file) for file in s3_files)
    st.sidebar.error(os.getenv('S3_BUCKET_NAME'))
    import boto3

    # Crea un cliente de S3
    s3 = boto3.client('s3')

    # Nombre del bucket
    bucket_name = os.getenv('S3_BUCKET_NAME')

    # Nombre del archivo
    file_name = 'GRI 1_ Fundamentos 2021 - Spanish.pdf'

    # Intenta obtener el archivo
    try:
        s3.head_object(Bucket=bucket_name, Key=file_name)
        print("File exists and you have permission to access.")
    except Exception as e:
        print(e)
except Exception as e:
    st.sidebar.error("Error al obtener los ficheros de S3.")
    s3_files_str = str(e)

# Mostrar la lista de ficheros de S3
st.sidebar.text_area("Ficheros de S3:", value=s3_files_str, height=100)

try:
    embed_pdf.embed_all_docs()
    st.sidebar.info("Ación realizada!")
except Exception as e:
    st.sidebar.error(e)
    st.sidebar.error("Failed to embed documents.")


# create the app
st.title("Bienvenidos al asistente del observatorio Recava de la UCLM")

# load the agent
from llm_helper import convert_message, get_rag_chain, get_rag_fusion_chain

rag_method_map = {
    'Basic RAG': get_rag_chain,
    'RAG Fusion': get_rag_fusion_chain
}
chosen_rag_method = st.sidebar.radio(
    "Choose a RAG method", rag_method_map.keys(), index=0
)
get_rag_chain_func = rag_method_map[chosen_rag_method]

# create the message history state
if "messages" not in st.session_state:
    st.session_state.messages = []

# render older messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# render the chat input
prompt = st.chat_input("Introduzca su pregunta...")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})

    # render the user's new message
    with st.chat_message("user"):
        st.markdown(prompt)

    # render the assistant's response
    with st.chat_message("assistant"):
        retrival_container = st.container()
        message_placeholder = st.empty()

        retrieval_status = retrival_container.status("**Context Retrieval**")
        queried_questions = []
        rendered_questions = set()
        def update_retrieval_status():
            for q in queried_questions:
                if q in rendered_questions:
                    continue
                rendered_questions.add(q)
                retrieval_status.markdown(f"\n\n`- {q}`")
        def retrieval_cb(qs):
            for q in qs:
                if q not in queried_questions:
                    queried_questions.append(q)
            return qs
        
        # get the chain with the retrieval callback
        for file in s3_files:
            custom_chain = get_rag_chain_func(file, retrieval_cb=retrieval_cb)
        
            if "messages" in st.session_state:
                chat_history = [convert_message(m) for m in st.session_state.messages[:-1]]
            else:
                chat_history = []

            full_response = ""
            for response in custom_chain.stream(
                {"input": prompt, "chat_history": chat_history}
            ):
                if "output" in response:
                    full_response += response["output"]
                else:
                    full_response += response.content

                message_placeholder.markdown(full_response + "▌")
                update_retrieval_status()

            retrieval_status.update(state="complete")
            message_placeholder.markdown(full_response)

        # add the full response to the message history
        st.session_state.messages.append({"role": "assistant", "content": full_response})