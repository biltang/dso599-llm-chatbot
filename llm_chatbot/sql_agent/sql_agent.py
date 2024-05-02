import re

from langchain.tools import tool
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.utilities import OpenWeatherMapAPIWrapper, SQLDatabase
from langchain_openai import OpenAI, ChatOpenAI
from langchain_core.language_models.base import BaseLanguageModel

@tool
def celsius_to_fahrenheit_conversion(celsius_tmp: str) -> str:
    """Converts Celsius temperature to Fahrenheit temperature.

    Args:
        celsius_tmp (str): temperature in Celsius to convert

    Returns:
        str: temperature in Fahrenheit
    """
    celsius_tmp = re.sub(r'[^0-9.-]', '', celsius_tmp)
    fahrenheit_tmp = (float(celsius_tmp) * 9/5) + 32
    
    return str(fahrenheit_tmp)


def get_sql_agent(llm: BaseLanguageModel, db_path: str, agent_type: str="openai-tools"):
    """Creates an sql agent using the given language model and database path.

    Args:
        llm (BaseLanguageModel): large language model instance
        db_path (str): path to the sqlite database
        agent_type (str, optional): agent type. Defaults to "openai-tools".

    Returns:
        Langchain AgentExecutor: an sql agent executor
    """
    db = SQLDatabase.from_uri(f"sqlite:///{db_path}")
    agent_executor = create_sql_agent(llm, db=db, agent_type=agent_type, verbose=True)
    return agent_executor


@tool
def query_dino_name_tool(dino_id: str) -> str:
    """Queries the sql database in dinosaur table for the dinosaur name when given
    a dinosaur ID and returns the name. 
    
    We create an explicit tool function initializing a sql_agent so that an agent_executor
    can use this as a tool function in the pipeline. Useful when we want a central react agent
    to use this tool function. 

    Args:
        dino_id (str): ID of the dinosaur to query

    Returns:
        str: Sentence with the dinosaur name
    """
    
    # ------------------------------------------------------------------------------
    # Creates a sql agent using an instance of ChatOpenAI model and a sqlite database
    # ------------------------------------------------------------------------------
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0) # chat model
    db_path = '../sqllite_dino_db/dino_name.db' # path to the sqlite database
    sql_agent = get_sql_agent(llm, db_path=db_path)
    # ------------------------------------------------------------------------------
    
    # use the sql agent to query the database
    response = sql_agent.invoke(f"What is the name of dinosaur with ID {dino_id}?")
    
    return response['output']