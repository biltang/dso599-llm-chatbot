import boto3
from botocore.exceptions import ClientError
import csv
from langchain.tools import tool, Tool


def check_table_exists(table_name: str) -> bool:
    """Check if a DynamoDB table exists with the given name.

    Args:
        table_name (str): table name to check

    Returns:
        bool: True if the table exists, False otherwise
    """
    dynamodb = boto3.client('dynamodb')
    
    try:
        response = dynamodb.describe_table(TableName=table_name)
        print(f"Table '{table_name}' exists. Table status: {response['Table']['TableStatus']}")
        return True
    except ClientError as e:
        if e.response['Error']['Code'] == 'ResourceNotFoundException':
            print(f"Table '{table_name}' does not exist.")
            return False
        else:
            raise  # Propagates other unexpected exceptions


def create_dynamodb_table(table_name: str, key_schema: dict, attribute_definitions: dict, provisioned_throughput: dict):
    """Create a DynamoDB table with the given parameters.

    Args:
        table_name (str): table name to create
        key_schema (dict): schema for the key
        attribute_definitions (dict): attribute definitions
        provisioned_throughput (dict): provisioned throughput parameters

    Returns:
        boto3.resources.factory.dynamodb.Table: DynamoDB table object
    """
    dynamodb = boto3.resource('dynamodb')
    
    # create table
    table = dynamodb.create_table(
        TableName=table_name,
        KeySchema=key_schema,
        AttributeDefinitions=attribute_definitions,
        ProvisionedThroughput=provisioned_throughput
    )
    table.wait_until_exists()  # Wait until the table is created
    
    return table


def read_csv_and_upload_to_dynamodb(table_name: str, csv_file_path: str):
    """Read CSV file and upload the data to the given DynamoDB table.

    Args:
        table_name (str): DynamoDB table name to upload data
        csv_file_path (str): path to the CSV file
    """
    dynamodb = boto3.resource('dynamodb')
    table = dynamodb.Table(table_name)
    
    # Read CSV file and upload data to DynamoDB table
    with open(csv_file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # Transform data if necessary
            table.put_item(Item=row)


def query_dynamodb_dino_tbl_by_date(date: str) -> list:
    """Query DynamoDB table by date to extract relevant information about dino transportation such as
    Dyno ID and City name.

    Args:
        date (str): date to query

    Returns:
        list: rows with the given date
    """
    table_name = 'dino_info'
    dynamodb = boto3.resource('dynamodb')
    table = dynamodb.Table(table_name)

    response = table.query(
        KeyConditionExpression='#date = :date',
        ExpressionAttributeNames={
            '#date': 'Date'  # Substitute for the reserved keyword 'Date'
        },
        ExpressionAttributeValues={
            ':date': date
        }
    )

    return response['Items']


def query_dynamodb_dino_tbl_by_date_tool():
    """ Create a langchain Tool object for querying DynamoDB table by date to extract relevant information about dino/dinosaur
    """
    return Tool.from_function(
            func=query_dynamodb_dino_tbl_by_date,
            name="query_dynamodb_dino_tbl_by_date",
            description="""Query DynamoDB table by date to extract relevant information about dino/dinosaur 
                            transportation such as
                            Dyno/Dino ID and City name.""",
            input_arg='date',
            output_arg='dino_info'
        )
    
    
def create_dino_dynamo_table(table_name: str, csv_file_path: str):
    """Create a DynamoDB table for dino transportation information and upload data from CSV.

    Args:
        table_name (str): table name to create in DynamoDB
        csv_file_path (str): path to the CSV file containing dino transportation information
    """

    key_schema = [
        {'AttributeName': 'Date', 'KeyType': 'HASH'},  # Partition key
        {'AttributeName': 'DinoID_Transported', 'KeyType': 'RANGE'}  # Sort key
    ]

    attribute_definitions = [
        {'AttributeName': 'Date', 'AttributeType': 'S'},
        {'AttributeName': 'DinoID_Transported', 'AttributeType': 'S'}
    ]
    
    provisioned_throughput = {
        'ReadCapacityUnits': 5,
        'WriteCapacityUnits': 5
    }

    # Create table
    create_dynamodb_table(table_name, key_schema, attribute_definitions, provisioned_throughput)

    # Upload data from CSV
    csv_file_path = '../../csv_data/dino_records.csv'
    read_csv_and_upload_to_dynamodb(table_name, csv_file_path)
    
    
def main():
    
    table_name = 'dino_info' # Table name to create in DynamoDB
    csv_file_path = '../../csv_data/dino_records.csv' # Path to the CSV file
    
    create_dino_dynamo_table(table_name, csv_file_path)


if __name__ == '__main__':
   main()