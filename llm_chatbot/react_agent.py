from dotenv import load_dotenv
import os 
import re
import ast

from langchain.tools import tool
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent, load_tools
from langchain_openai import OpenAI, ChatOpenAI
import openai
from sql_agent.sql_agent import celsius_to_fahrenheit_conversion, query_dino_name_tool
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import BaseTool, StructuredTool, tool

from vector_store.vector_store import create_retriver_tool_chroma_db
from aws_features.dynamo_db import query_dynamodb_dino_tbl_by_date_tool

# Load environment variables from .env file
load_dotenv()

    
@tool
def dino_temp_safety_check(temperature_list: str) -> str:
    """Given a string formatted list of temperatures, where the first value is the current temperature,
    the second value is the safe temperature range low value,
    the third value is the safe temperature range high value, 
    this tool will return whether it is safe for the dinosaurs at the current temperature. The input 
    must be three different arguments. The output will be a string.

    Args:
        temperature_list (str): A string formatted list of temperatures delimited by commas.

    Returns:
        str: message indicating if it is safe for the dinosaurs at the current temperature.
    """
    current_temperature, temperature_range_low, temperature_range_high = temperature_list.split(",") # Split the string into a list of temperatures
    
    current_temperature = float(re.sub(r'[^0-9.-]', '', current_temperature)) # Remove any non-numeric characters from the current temperature
    temperature_range_low = float(re.sub(r'[^0-9.-]', '', temperature_range_low)) # Remove any non-numeric characters from the low temperature range
    temperature_range_high = float(re.sub(r'[^0-9.-]', '', temperature_range_high)) # Remove any non-numeric characters from the high temperature range
    
    if current_temperature < temperature_range_low:
        return "It is not safe for the dinosaurs at this temperature because it is too cold."
    elif current_temperature > temperature_range_high:
        return "It is not safe for the dinosaurs at this temperature because it is too hot."
    else:
        return "It is safe for the dinosaurs at this temperature."
    

def get_react_agent(tools, llm, prompt: str=None, hub_prompt: str="hwchase17/react"):
    """ Instantiates a react agent with the given tools and llm. If a prompt is not given, it will pull the prompt from the hub.

    Args:
        tools: a list of tools to be used by the agent.
        llm: the language model to be used by the agent.
        prompt (str, optional): user supplied prompt. If None/null, defaults to using the hub_prompt for react agent. Defaults to None.
        hub_prompt (str, optional): prompt template to pull from the langchain hub. Defaults to "hwchase17/react".

    Returns:
        a react agent
    """
    # Pull the prompt from the hub if not given
    if prompt is None:
        prompt = hub.pull(hub_prompt)
        
    return create_react_agent(llm, tools, prompt)


def load_existing_tools(load_api_keys: dict = {}) -> list:
    """Loads existing langchain tools and sets the api keys if provided.

    Args:
        load_api_keys (dict, optional): api keys needed for loading langchain existing tools. Defaults to {}.

    Returns:
        list: list of tools to be used by the agent
    """
    # Set the api keys as environment variables
    for key, value in load_api_keys.items():
        os.environ[key] = value
        
    tools = load_tools(["openweathermap-api"]) # Load the existing tools
    
    return tools
    
    
def get_all_tools(load_api_keys: dict = {}, chroma_db_path='../chroma_vector_store/'):
    """Gets all the tools needed for the agent.

    Args:
        load_api_keys (dict, optional): supplies api keys needed to load preexisting langchain tools. Defaults to {}.
        chroma_db_path (str, optional): path to chromadb vector store for running rag agent. Defaults to '../chroma_vector_store/'.

    Returns:
        list: list of tools to be used by the agent
    """
    tools = load_existing_tools(load_api_keys) # Load the existing tools
    
    tools = tools + [create_retriver_tool_chroma_db(db_path=chroma_db_path), # Add the chroma db tool for rag retrieval
                     query_dynamodb_dino_tbl_by_date_tool(), # Add the dynamodb tool for querying the dino table
                     celsius_to_fahrenheit_conversion, # Add the celsius to fahrenheit conversion tool
                     query_dino_name_tool, # Add the dino name query tool from sqlite table
                     dino_temp_safety_check, # Add the dino temperature safety check tool
                     dino_temp_safety_check_workflow_tool] # Add the dino temperature safety check end-to-end workflow tool
    return tools


# OpenAI Agent
def OpenAI_agent(model: str, tools: list, prompt: str=None, hub_prompt: str="hwchase17/react", temperature: float=0.7) -> AgentExecutor:
    """ Creates an OpenAI agent with the given model, tools, and prompt. If a prompt is not given, it will pull the prompt from the hub.

    Args:
        model (str): OpenAI model name.
        tools (list): list of tools to be used by the agent.
        prompt (str, optional): user supplied agent prompt. If none, will default to hub_prompt. Defaults to None.
        hub_prompt (str, optional): a prompt to use from langchain hub. Defaults to "hwchase17/react".
        temperature (float, optional): model temperature. Defaults to 0.7.

    Returns:
        AgentExecutor: an agent executor object for the OpenAI agent.
    """
    
    openai.api_key = os.getenv("OPENAI_API_KEY")
    
    llm = ChatOpenAI(model_name=model, temperature=temperature)

    agent = get_react_agent(tools=tools, llm=llm, prompt=prompt, hub_prompt=hub_prompt)

    # Create an agent executor by passing in the agent and tools
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    return agent_executor



def agent_check_safety_actions(agent_executor: AgentExecutor, context_info: str):
    """ Given the context information, this function checks what specific actions we can take to keep the relevant dinosaur safe based on the existing documentation.
    
    Args:
        agent_executor (AgentExecutor): an agent executor object for the OpenAI agent.
        context_info (str): context information to be used by the agent.
        
    Returns:
        str: the actions to take to keep the relevant dinosaur safe based on the existing documentation.
    """
    input_msg = f"""Given this information: {context_info}.
                Check what specific actions we can take to to keep the relevant dinosaur safe based on the existing documentation.
                (Make sure to reference the correct document since you are given the dinosaur name)"""
    action_response = agent_executor.invoke({"input": f"{input_msg}. Only use a tool if needed, otherwise respond with Final Answer"})
    return action_response['output']


@tool
def dino_temp_safety_check_workflow_tool(date):
    """Given a date, this tool implements an end-to-end workflow to check the safety of 
    transporting a dinosaur/dino on that date and crafts a email to the manager.
    Given the date, it finds out the dino/dinosaur being transported and information
    about the temperature to check if it is safe, and if not, what actions to take to keep the dino safe.
    Given this information, it then crafts a status email to the manager, which should then be
    passed back to the user"""
    
    load_dotenv()
    api_keys = {'OPENWEATHERMAP_API_KEY':os.getenv("OPENWEATHERMAP_API_KEY")}
    tools = get_all_tools(load_api_keys=api_keys)
    agent_executor = OpenAI_agent(model='gpt-4',
             tools=tools, 
             prompt=None, 
             hub_prompt="hwchase17/react", 
             temperature=0)
    
    extra_context = "Only use a tool if needed, otherwise respond with Final Answer"
        
    # Get the safe temperature range for the dinosaur transported on date
    input_msg = f"""What is the safe temperature range of the dinosaur being transported on date {date}? 
                    In your response, also give the city and name of the dinosaur as well."""
    temperature_response = agent_executor.invoke({"input": f"{input_msg}. {extra_context}"})['output']
  
    
    # Get current temperature of the city of transport
    input_msg = f"""Given the following context: 
                {temperature_response}
                what is the current temperature in fahrenheight of the referenced city? 
                Give the answer like this example: 'The current temperature in CITY is TEMPERATURE"""
    current_temp = agent_executor.invoke({"input": f"{input_msg}. {extra_context}"})['output']
    
    # is the current temperature safe for the dinosaur
    input_msg = f"""{temperature_response + ' ' + current_temp}
                    Is it safe to transport the dinosaur at that temperature? You do not need to do any temperature conversion."""
    safety_response = agent_executor.invoke({"input": f"{input_msg}. {extra_context}"})['output']
    
    # if the temperature is not safe - find out what actions to take
    input_msg = f"""Given this information: {temperature_response + ' ' + current_temp + ' ' + safety_response}
                If it is not safe to transport the dinosaur,
                check what specific actions we can take to to keep the relevant dinosaur safe based on the existing documentation. 
                (Make sure to reference the correct document since you are given the dinosaur name)"""
    action_response = agent_executor.invoke({"input": f"{input_msg}. {extra_context}"})['output']
    
    # craft a status email to the manager
    message_info = temperature_response + ' ' + current_temp + ' ' + safety_response + ' ' + action_response
    input_msg = f"""Given this information: {message_info}
                Craft a status email to inform the manager about the dinosaur. It should look like:
                Subject: ...
                
                Dear Manager,
                ...
                Best
                Your Name.
                Your Final Answer should be the email."""
    email = agent_executor.invoke({"input": f"{input_msg}. {extra_context}"})['output']
    
    return email
