import streamlit as st
import os
import embed_pdf
from llm_helper import convert_message, get_rag_chain

# get openai api key from environment variable
openapi_key = os.getenv("OPENAPI_KEY")

# create the app
st.title("Bienvenidos al asistente del observatorio Recava de la UCLM")

# Ask the user to upload a PDF file
uploaded_file = st.file_uploader("Sube un archivo PDF para retrieval", type="pdf")

# If a file has been uploaded
if uploaded_file is not None:
    # Read the file
    file_bytes = uploaded_file.read()

    # Call the method to save the file to MongoDB and generate index
    search_index = embed_pdf.embed_document(file_bytes, uploaded_file.name)
    st.session_state['search_index'] = search_index

    # Store the file in session state for later use
    st.session_state['pdf_file'] = file_bytes

    st.success("Archivo PDF subido y almacenado para su uso posterior.")

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

        # Add a placeholder for the loading message
        loading_message = st.empty()
        loading_message.text("Estamos procesando su pregunta...")

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
        custom_chain = get_rag_chain(retrieval_cb=retrieval_cb, vectorstore=search_index)
    
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

        # Once the response is ready, clear the loading message
        loading_message.empty()

        # add the full response to the message history
        st.session_state.messages.append({"role": "assistant", "content": full_response})