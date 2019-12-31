import boto3
sqs = boto3.resource('sqs')
client = boto3.client('sqs')


# SQS 
def populate_events_queue(queue, model, events, dryrun=True):
    print(f'Queue = {queue}')
    '''
    Populate queue on AWS SQS for production or test runs

    Options:
        1. events as a list [1, 19, 145] (will create only events in the list)
        2. events as an interger 300 (will create events from 1-300)
    '''
    if isinstance(events, list):
        for event in events:
            msg = f'{model}-E{event:04}'
            if dryrun:
                print(msg)
            else:
                response = queue.send_message(MessageBody=msg,
                                      MessageGroupId=msg,
                                      MessageDeduplicationId=msg)

                print(msg, response.get('MessageId'))
            
    elif isinstance(events, int):
        for event in range(1, events+1):
            msg = f'{model}-E{event:04}'
            if dryrun:
                print(msg)
            else:
                response = queue.send_message(MessageBody=msg,
                                      MessageGroupId=msg,
                                      MessageDeduplicationId=msg)

                print(msg, response.get('MessageId'))
    else:
        return print('Check inputs')


def print_events_in_queue(queue):
    live_messages=[]
    messages = client.receive_message(QueueUrl=queue.url,MaxNumberOfMessages=10)
    for message in messages['Messages']:
        live_messages.append(message['Body'])
        print(message['Body'])
        

def purge_events_queue(queue):
    queue.purge()
    

    
# S3