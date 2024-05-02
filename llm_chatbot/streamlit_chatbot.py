import os 
from collections.abc import Callable

import streamlit as st
import hydra
from omegaconf import DictConfig
from langchain_openai import OpenAI, ChatOpenAI
from langchain.agents import AgentExecutor

#from llms import load_llm_pipeline
from react_agent import get_all_tools, get_react_agent
from vector_store.vector_store import create_retriver_tool_chroma_db, create_chroma_db_vector_store
from aws_features.dynamo_db import check_table_exists, create_dino_dynamo_table
# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()


# Define a function to add a new message pair (user and bot) to the history
def add_to_history(user_message: str, bot_response: str):
    """ Check for message history and append the new message pair to the history.

    Args:
        user_message (str): current user message
        bot_response (str): current bot response
    """
    
    # Check if message history exists in the session state
    if 'message_history' not in st.session_state:
        # Initialize message history
        st.session_state.message_history = []
    
    # Append the new message pair to the history
    st.session_state.message_history.append((user_message, bot_response))
    
    # Ensure only the last three pairs are kept
    if len(st.session_state.message_history) > 3:
        st.session_state.message_history = st.session_state.message_history[-3:]
        

def display_history():
    """ Display the message history in the app.
    """
    
    if 'message_history' in st.session_state: # check if message history exists in the session state
        st.write('Message History:')
        for index, message_pair in enumerate(st.session_state.message_history): # loop through the message history
            user_message, bot_response = message_pair # get the user message and bot response
            st.write('User:', user_message)
            st.write('Bot:', bot_response)
            st.write('---')
            

def initialize_data_sources(chromadb_args: dict, dynamodb_args: dict):
    """ Initialize data sources for the chatbot application.
    
    Args:
        chromadb_args (dict): dictionary of arguments for initializing the chroma db
        dynamodb_args (dict): dictionary of arguments for initializing the dynamo db
    """
    
    # ChromaDB vector store for dino safety pdf info - part 2
    if not os.path.exists(chromadb_args['db_path']):
        create_chroma_db_vector_store(**chromadb_args)
    else:
        print("Chroma DB already exists.")
        
    # Initialize DynamoDB for dino transport info - part 3
    if not check_table_exists(dynamodb_args['table_name']):
        create_dino_dynamo_table(**dynamodb_args)
        
    
def initialize_rag(model: str, rag_function: Callable, prompt: str, hub_prompt: str, db_path: str, temperature=0.7):
    """ Initialize the RAG model.

    Args:
        model (str): model id
        rag_function (Callable): rag function to use
        prompt (str): user prompt to use. If none, defaults to hub_prompt
        hub_prompt (str): hub prompt to use
        db_path (str): path to the chroma db for rag retrieval
        temperature (float, optional): model temperature. Defaults to 0.7.

    Returns:
        langchain agent executor
    """
    rag_func = hydra.utils.instantiate(rag_function) # instantiate the rag function
    tools = get_all_tools(chroma_db_path=db_path) # get all tools from the chroma db
    
    # Initialize the agent executor
    agent_executor = rag_func(model=model,
                            tools=tools,
                            prompt=prompt,
                            hub_prompt=hub_prompt,
                            temperature=temperature)
    return agent_executor


@hydra.main(version_base=None, config_path="conf", config_name="streamlit_config")
def main(cfg: DictConfig):
    """ Main function to run the Streamlit chatbot application.
    
    Args:
        cfg (DictConfig): Hydra configuration object. We use hydra config to load yaml configuration files.
    """
    
    # check if data sources are initialized, and if not, initialize them
    initialize_data_sources(chromadb_args={'doc_dir': '../pdf_data/', 
                                           'db_path': '../chroma_vector_store/'}, 
                            dynamodb_args={'table_name': 'dino_info',
                                           'csv_file_path': '../csv_data/dino_records.csv'}) 
    
    st.title(cfg.llm_backend.chatbot_title)

    with st.form(key='my_form'):
        # create two columns
        col1, col2 = st.columns(2)
        
        with col1:
            # ---------------------------------- 
            # col1 - LLM Temperature Value - Radio buttons
            # ----------------------------------
            options = [0.0, 0.4, 0.9]
            temp_selection = st.radio("Choose a LLM temperature value:", options)
            st.write(f'You selected temperature value: {temp_selection}')
    
        with col2:
            # ----------------------------------
            # col2 - LLM Options - Selectbox
            # ----------------------------------
            llm_options = list(cfg.llm_backend.llm_options.keys())
            llm_selection = st.selectbox(
                            'Which llm would you like to use?',
                            llm_options)
            st.write('You selected:', llm_selection)

            # ----------------------------------
            # Initialize LLM
            # ----------------------------------
            if cfg.rag.rag_application: # if using RAG
                
                # Initialize RAG
                llm_func = initialize_rag(model=cfg.llm_backend.llm_options[llm_selection].model_id,
                                          rag_function=cfg.llm_backend.llm_options[llm_selection].function_call,
                                          prompt=cfg.rag.prompt,
                                          hub_prompt=cfg.rag.hub_prompt,
                                          db_path=cfg.rag.db_path,
                                          temperature=temp_selection)
                
            else: # if not using RAG, instantiate the LLM
                
                # Instantiate the LLM
                llm_func = hydra.utils.instantiate(cfg.llm_backend.llm_options[llm_selection].function_call)
                model_id = cfg.llm_backend.llm_options[llm_selection].model_id
                model_input = model_id
                    
        # ----------------------------------
        # input box for user to type in
        # ----------------------------------
        user_input = st.text_input("Type your message:")
        submit_button = st.form_submit_button(label='Submit')
        
        if submit_button: # if the submit button is clicked
            if user_input: # if user input is not empty
                
                # process model response differently depending on if RAG is used
                if cfg.rag.rag_application:
                    
                    # for rag application, check the output of the agent executor
                    response = llm_func.invoke({"input": f"""You are a helpful assistant, 
                                                             help me answer the follow: {user_input}. 
                                                             Only use a tool if needed, otherwise use your
                                                             original knowledge
                                                             and respond with Final Answer."""})
                    response = response['output']
                    
                else:
                    
                    # for non-rag application, directly use response
                    response = llm_func(model=model_input,
                                        prompt=user_input,
                                        temperature=temp_selection)
                
                
                add_to_history(user_input, response) # add user input and bot response to message history
                st.session_state.user_input = "" # clear the user input box
                
            else: # if user input is empty, display a message to prompt user to enter a message
                st.write('Please enter a question or statement to interact with the AI.')
                
    # Display the message history
    display_history()
            

if __name__ == '__main__':
   main()
   



