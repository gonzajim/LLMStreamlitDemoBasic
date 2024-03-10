import streamlit as st

from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

from llm_helper import get_agent_chain, get_lc_oai_tools

with st.sidebar:
    openai_api_key = st.secrets["OPENAI_API_KEY"]

    "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"
    "[View the source code](https://github.com/streamlit/llm-examples/blob/main/pages/2_Chat_with_search.py)"
    "[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/streamlit/llm-examples?quickstart=1)"

    default_text = ""
    user_instruction_text = st.text_area(
        "Instrucciones para el usuario",
        default_text,
    )
st.title("🔎 UCLM - RECAVA RAG Chatbot")

"""
Este chat está creado con streamlit + langchain + openai + mongodb
"""

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hola, soy el asistente del observatorio de la UCLM, en qué puedo ayudarte?"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input(placeholder="Puedes decirme algún ejemplo de ayuda que puedes proporcionar?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    if not openai_api_key:
        st.info("Por favor, añade tu clave de OPENAI para continuar.")
        st.stop()

    llm = ChatOpenAI(model_name="gpt-3.5-turbo-1106", openai_api_key=openai_api_key, streaming=True)
    lc_tools, _ = get_lc_oai_tools()
    search_agent = initialize_agent(lc_tools, llm, agent=AgentType.OPENAI_FUNCTIONS, handle_parsing_errors=True, verbose=True)

    agent_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", user_instruction_text),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    search_agent.agent.prompt = agent_prompt
    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        response = search_agent.run(prompt, callbacks=[st_cb])
        # search_agent = get_agent_chain(callbacks=[st_cb])
        # response = search_agent.invoke({"input": prompt})
        # response = response["output"]
        
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.write(response)
