import sys 
import argparse
from dotenv import load_dotenv
import os

from langchain.tools import tool

from react_agent import OpenAI_agent, get_all_tools
from aws_features.aws_utils import aws_send_sms


def dino_temp_safety_check_workflow(model_name: str, temperature: float, date: str, phonenumber: str, api_keys: dict, extra_context: str=None) -> str:
    """ This function will run the workflow for the dinosaur temperature safety check.
    
    Args:
        model_name (str): The name of the model to use for the chatbot
        temperature (float): The temperature to use for the chatbot
        date (str): The date of transport
        phonenumber (str): The phone number to send the email to
        api_keys (dict): The api keys to use for the chatbot
        extra_context (str): The extra context to use for the chatbot
    
    Returns:
        str: The email that was sent to the manager
    """
    
    # Get all the tools
    tools = get_all_tools(load_api_keys=api_keys)
    
    # Create the agent executor
    agent_executor = OpenAI_agent(model=model_name,
             tools=tools, 
             prompt=None, 
             hub_prompt="hwchase17/react", 
             temperature=temperature)
    
    # Set the extra context if not provided
    if extra_context is None:
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
    
    # send the email through aws
    sms_response = aws_send_sms(phonenumber, email)
    
    if sms_response.get('MessageId') and sms_response['ResponseMetadata']['HTTPStatusCode'] == 200:
        print("Message sent successfully!")
    else:
        print("Failed to send message.")
    
    return email


def main(args):
    
    print(f"""model name {args.modelname}, model temperature {args.temperature}, transport date {args.date}, phone number {args.phonenumber}""")
    
    # load the api keys
    load_dotenv()
    api_keys = {'OPENWEATHERMAP_API_KEY':os.getenv("OPENWEATHERMAP_API_KEY")}
    
    email = dino_temp_safety_check_workflow(model_name=args.modelname, 
                                    temperature=args.temperature, 
                                    date=args.date, 
                                    phonenumber=args.phonenumber, 
                                    api_keys=api_keys)
    
    print(email)
    

if __name__ == '__main__':
   # Create the parser
    parser = argparse.ArgumentParser(description="In take some arguments.")
    
    # Add arguments
    parser.add_argument('--modelname', type=str, default='gpt-3.5-turbo-0125',
                        help='OpenAI model name to use for the chatbot') 
    
    parser.add_argument('--temperature', type=str, default=0,
                        help='Model temperature to use for the chatbot')
    
    # Add arguments
    parser.add_argument('--date', type=str, default='3/19/2024',
                        help='Date of transport')
    
    # Add arguments
    parser.add_argument('--phonenumber', type=str, default='+16319353837',
                        help='phone number to send the email to')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Call the main function with parsed arguments
    main(args)
    
