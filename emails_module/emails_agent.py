###### Email Assistant agent
    # This file contains the email agent
    # Will be imported into the main graph

##### General setup
#### Import libraries
### General libraries
import os, operator, base64
from dotenv import load_dotenv
from utilities import *
from typing import Optional
### Model libraries
from langchain_google_genai import ChatGoogleGenerativeAI 
from langchain_cohere import CohereEmbeddings
### Storage libraries
from langgraph.store.postgres import PostgresStore
from psycopg import Connection
from langgraph.checkpoint.sqlite import SqliteSaver
### Tool libraries
from pydantic import BaseModel, Field, field_validator
from langmem import create_manage_memory_tool, \
    create_search_memory_tool
from langchain_core.tools import tool
### Gmail API libraries
from google.oauth2.credentials import Credentials  
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from email.message import EmailMessage

##### Miscelanous utility functions
def store_items(
    store: PostgresStore, 
    namespace: str, 
    item_name: str,
    item: dict
):
    check = store.get(namespace, item_name)
    if check is None:
        store.put(
            namespace,
            item_name,
            item
        )

def retrieve_items(
    store: PostgresStore,
    namespace: str,
    items_names: list,
):
    items = {}
    for name in items_names:
        to_store = store.get(
            namespace,
            name,
        )
        items[name] = to_store.value
    
    return items


#### Unpacking env variables
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")
cohere_api_key = os.getenv("COHERE_API_KEY")
postgres_conn = os.getenv("POSTGRES_CONN")


#### Setup models
### Base LLM
base_llm = ChatGoogleGenerativeAI(
    api_key = google_api_key,
    model = "gemini-2.5-flash-lite",
)

### Base embedding model
# embedding_model = CohereEmbeddings(
#     cohere_api_key=cohere_api_key,
#     model = "embed-english-light-v3.0",
# )

#### Setting up DB storage utility
DB_NAME = "output.sqlite"
TABLE_NAME = "email_assistant"
storage = Storage(DB_NAME, TABLE_NAME)


#### Creating memories
### Short-term memories
conn = sqlite3.connect('checkpoints.sqlite', check_same_thread=False)
    #"check_same_thread = False" => enables multi-thread usage
memory = SqliteSaver(conn)

### Long-term memory
## Postgres Store
conn2 = Connection.connect(postgres_conn, autocommit=True)
    #"postgresql://user:pass@localhost:5432/dbname"
store = PostgresStore(
    conn2,
    # index={"embed": embedding_model, "dims": 384}
)
store.setup()
    # This is needed in order to capture the tables schema needed in the database


### Setup profile
profile = {
    "name": "Miguel",
    "full_name" : "Miguel Gomez",
    "user_profile_background": "Programmer"
}





























#### Prompt instructions
### Setup prompt instructions
instructions ={
    "router_rules": {
        "ignore": "Marketing newsletters, spam emails, mass company announcements",
        "notify": "Team member out sick, build system notifications, project status updates",
        "respond": "Direct questions from team members, meeting requests, critical bug reports",
    },
    "emailing_agent": "Use these tools when sending or receiving emails",
    "calendar_agent":"Use these tools when working with the calendar, directly or indirectly",
    "memory_agent": "Use these tools when updating important information retrieved from the other agents"
}


### Storing instructions into long-term memory
namespace = ("email_agent", "instructions")
for key, value in instructions.items():
    print(f'KEY: {key}')
    if isinstance(value, dict): continue
        # Skipping router_rules, will do it manually
    store_items(store, namespace, key, value)

for key, value in instructions["router_rules"].items():
    store_items(
        store,
        namespace,
        f"router_{key}",
        value
    )


























##### Tools
#### Memory tools
### Manage memory tool
manage_memory_tool = create_manage_memory_tool(
    namespace=(
        "email_agent",
        "semantic_memory"
    )
)

### Search memory tool
search_memory_tool = create_search_memory_tool(
    namespace=(
        "email_agent",
        "semantic_memory"
    )
)

#### General functions for email and calendar agents
### General scope
SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.compose",
    "https://www.googleapis.com/auth/calendar.events.owned",
]
### Gathering credentials
def gather_credentials():
    creds = None
    # The file token.json stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.
    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                "credentials.json", SCOPES
            )
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open("token.json", "w") as token:
            token.write(creds.to_json())
    return creds


#### Emailing tools
### Getting tools
# @tool
def get_emails(num_emails: int = 5) -> dict:
    creds = gather_credentials()
    to_return = {}
    try:
        service = build('gmail', 'v1', credentials=creds)
        results = (
            service.users().messages()
            .list(maxResults = num_emails, userId="me", labelIds=["INBOX"])
                #Note: technically "INBOX" shows "ALL MAIL" mail
            .execute()
        )

        messages = results.get("messages", [])
        if not messages:
            print("No messages found.")
            return to_return
    
    except HttpError as error:
        print(f"Error: {error}")

    for msg in messages:
        txt = (
            service.users().messages()
            .get(userId="me", id=msg["id"]).execute()
        )
        try:
            payload = txt['payload']
            headers = payload['headers']
            body = payload['body']
            parts = payload['parts']
        except: continue
        for data in headers:
            if data['name'] == 'Subject':
                to_return["subject"] = data['value']
            if data['name'] == "From":
                to_return["author"] = data['value']
            if data['name'] == 'To':
                to_return["to"] = data['value']
        encoded_body = parts[0]['body']['data']
        encoded_body = encoded_body.replace("-","+").replace("_","/")
            #Necessary to decode the email properly
        decoded_data = base64.b64decode(encoded_body)
        decoded_data = decoded_data.decode('utf-8')
        to_return['email_thread'] = decoded_data
        break
    
    return to_return

# print(get_emails(10))

#### Writing emails
# @tool
def write_email(
    to: str,
    subject: str,
    content: str,
) -> str:
    """Write and send an email"""
    creds = gather_credentials()
    try:
        service = build("gmail", "v1", credentials=creds)
        message = EmailMessage()

        message.set_content(content)

        message["To"] = to
        message["From"] = "jparra2357@gmail.com"
        message["Subject"] = subject

        # encoded message
        encoded_message = base64.urlsafe_b64encode(message.as_bytes()).decode()

        create_message = {"raw": encoded_message}
        send_message = (
            service.users()
            .messages()
            .send(userId="me", body=create_message)
            .execute()
        )
        print(f'Message Id: {send_message["id"]}')
    except HttpError as error:
        print(f"An error occurred: {error}")
    return f"Email sent to {to} with subject '{subject}'"


#### Calendar tools
### Pydantic model
class TimeAvailability(BaseModel):
    start: str = Field(
        description="Moment at which to start calendar check (format = 'YYYY-MM-DDTHH:MM:SS+/-HH:MM')"
    )
    end: str = Field(
        description="Moment at which to end calendar check (format = 'YYYY-MM-DDTHH:MM:SS+/-HH:MM')"
    )
    event_duration: Optional[int] = Field(
        default=30,
        description="Used to tell the duration of the event"
    )
    @field_validator("start", "end")
    def validate_time(cls, time: str):
        import re
        pattern = r"^20[2-9]{1}[0-9]{1}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\+|\-)?\d{2}:\d{2}$"
            # Patterns ensures year will be 2020+
        if not re.match(pattern, time):
            raise ValueError("time needs to be in ISO format string")
        return time










