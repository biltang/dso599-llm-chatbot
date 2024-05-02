import boto3
from botocore.config import Config


def aws_send_sms(phone_number: str, message: str) -> dict:
    """Send an SMS message using AWS SNS.

    Args:
        phone_number (str): phone number to send the SMS to
        message (str): message to send

    Returns:
        dict: response from the SNS service
    """
    # Create an SNS client
    sns_client = boto3.client('sns',
                              region_name='us-east-1')

    # Send SMS
    response = sns_client.publish(
        PhoneNumber=phone_number,  # include the country code, e.g., +1 for US numbers
        Message=message,
        MessageAttributes={
            'AWS.SNS.SMS.SMSType': {
                'DataType': 'String',
                'StringValue': 'Transactional'  # Use 'Promotional' for non-critical messages
            }
        }
    )
    return response
