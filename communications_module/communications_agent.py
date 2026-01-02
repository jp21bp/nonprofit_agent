###### Communications Assistant agent
    # This file contains the communications agent
    # Will be imported into the main graph

##### General setup
#### Import libraries
### General libraries
import os, operator, base64, sqlite3
from datetime import datetime, timedelta
from dotenv import load_dotenv
from typing import Optional, Literal, List
from typing_extensions import TypedDict, Literal, Annotated
### Langchain libraries
from langgraph.graph import StateGraph, START, END
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import AnyMessage, \
    SystemMessage, HumanMessage
### Model libraries
from langchain_google_genai import ChatGoogleGenerativeAI 
from langchain_cohere import CohereEmbeddings
from langgraph.types import Command
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
TABLE_NAME = "communications_assistant"
# storage = Storage(DB_NAME, TABLE_NAME)


#### Creating memories
### Short-term memories
# conn = sqlite3.connect('checkpoints.sqlite', check_same_thread=False)
#     #"check_same_thread = False" => enables multi-thread usage
# memory = SqliteSaver(conn)

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
    "user_profile_background": "Mid software engineer"
}





























#### Prompt instructions
### Setup prompt instructions
instructions ={
    "router_rules": {
        "communications": "Marketing newsletters, spam emails, mass company announcements",
        "calendar": "Team member out sick, build system notifications, project status updates",
        "memory": "Direct questions from team members, meeting requests, critical bug reports",
    },
    "communications_agent": "Use these tools when sending or receiving emails",
    "calendar_agent":"Use these tools when working with the calendar, directly or indirectly",
    "memory_agent": "Use these tools when updating important information retrieved from the other agents"
}


### Storing instructions into long-term memory
namespace = ("communications_agent", "instructions")
for key, value in instructions.items():
    print(f'KEY: {key}')
    if isinstance(value, dict): continue
        # Skipping router_rules, will do it manually
    store_items(store, namespace, key, {'instruction': f'{value}'})

for key, value in instructions["router_rules"].items():
    store_items(
        store,
        namespace,
        f"router_{key}",
        {'instruction': f'{value}'}
    )


























##### Tools
#### Memory tools
### Manage memory tool
manage_memory_tool = create_manage_memory_tool(
    namespace=(
        "communication_agent",
        "semantic_memory"
    )
)

### Search memory tool
search_memory_tool = create_search_memory_tool(
    namespace=(
        "communication_agent",
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
class ScheduleEvent(BaseModel):
    subject: str = Field(
        description='A name for the event to be created'
    )
    atendee_emails: list = Field(
        default = "jparra2357@gmail.com",
        description= 'A list of emails of all the participants in the event, including the one creating the event'
    )
    start: str = Field(
        description="Moment at which to start calendar event (format = 'YYYY-MM-DDTHH:MM:SS+/-HH:MM')"
    )
    end: str = Field(
        description="Moment at which to end calendar event (format = 'YYYY-MM-DDTHH:MM:SS+/-HH:MM')"
    )
    @field_validator("start", "end")
    def validate_time(cls, time: str):
        import re
        pattern = r"^20[2-9]{1}[0-9]{1}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\+|\-)?\d{2}:\d{2}$"
            # Patterns ensures year will be 2020+
        if not re.match(pattern, time):
            raise ValueError("time needs to be in ISO format string")
        return time


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

### Schedule Events
@tool(args_schema=ScheduleEvent)
def schedule_event(
    subject: str,
    attendee_emails: list, 
    start: str,
    end: str
) -> str:
    """Schdule a meeting on google calendar"""
    creds = gather_credentials()
    try: 
        service = build("calendar", "v3", credentials=creds)

        ### Create the events
        event = {
            'summary': subject,
            'start': {
                'dateTime': start,
                'timeZone': 'America/New_York',
            },
            'end': {
                'dateTime': end,
                'timeZone': 'America/New_York',
            },
            'attendees': [{'email': email} for email in attendee_emails],
        }

        event = service.events().insert(calendarId='primary', body=event).execute()
        print(f"Event created: {event.get('htmlLink')}")
    except HttpError as error:
        print(f"An error occured: {error}")
    return f"Meeting '{subject}' scheduled to start on {start}"


#### Checking avaialbility for a specific day
def check_day_availability(
    start: datetime, 
    end: datetime,  
    event_duration: int = 30,
) -> str:
    # Pydantic model can be done with field validator, see C17V
    creds = gather_credentials()
    day = start.strftime("%b %d")
    try:
        service = build("calendar", "v3", credentials=creds)
        events_result = (
            service.events()
            .list(
                calendarId="primary",
                timeMin=start.isoformat(),
                timeMax=end.isoformat(),
                singleEvents=True,
                orderBy="startTime",
            )
            .execute()
        )
        events = events_result.get("items", [])
    
    except HttpError as error:
        print(f"Error occured: {error}")

    if not events:
        print(f"Available on {day} from 700 to 1700\n")
        return f"Available on {day} from 700 to 1700\n"
    
    duration_delta = timedelta(minutes  = event_duration)
    entry_time = 700
    exit_time = 1700
    add_entry, add_exit = True, True 
    available_times = {
        "start":[],
        "end": [],
    }

    for i, event in enumerate(events):
        ### Getting the start and end times of the planned event
        planned_event_start = event["start"].get("dateTime", event["start"].get("date"))
        planned_event_end = event["end"].get("dateTime", event["end"].get("date"))
        
        
        ### Transforming times into datetime iso format
        start_datetime = datetime.fromisoformat(planned_event_start)
        end_datetime = datetime.fromisoformat(planned_event_end)

        ### Adding the event duration to the already planned events
            # This yields the latest available to start the new event
            # Note: latest avaialbility for new event 
                    # = start time of planned event - new event's duration
        latest_availability = start_datetime - duration_delta
        latest_availability = int(
            f"{latest_availability.hour}{latest_availability.minute:0>2d}"
        )
        
        ### Calculating the earliest availability to start new event
            # NOte: earliest availability = end of the planned event
        earliest_avaialability = int(
            f"{end_datetime.hour}{end_datetime.minute:0>2d}"
        )
        
        ### Edge cases
        if latest_availability < entry_time and earliest_avaialability > exit_time:
            # This would be an event that takes up the whole day
            # Ex: 600 - 1800
            break
        elif latest_availability > exit_time or earliest_avaialability < entry_time:
            # IF the end of a meeting happens BEFORE 7AM, then 
                    # it wouldn't matter registering the start of 
                    # that meeting, since it would ALSO be before
                    # 7 AM
                # The same can be said if beginning of event is 
                        # AFTER 1700, end of day
            # Ex: meeting 5am - 630am
                # End @ 630 am => continue to next event
            # Ex: 1800 - 1900
                # Start @ 1800 > 1700 = > continue to next eventx
            continue
        elif latest_availability < entry_time and earliest_avaialability > entry_time:
            # This is the case for when the event starts before entry time
                    # but it ends after the entry time
            # EX: 600 - 800
            # In this case, we won't add 700 to available start times
                # Since the earliest start time starts at 800
            add_entry = False
            available_times["start"].append(earliest_avaialability)
        elif latest_availability < exit_time and earliest_avaialability > exit_time:
            # This is when an event starts before exit time but ends afterwards
            # EX: 1600 - 1800
            # In this case we won't add 1700 to available end times
                # Since the latest time for availbility if 1600 - meeting_duration
            add_exit = False
            available_times["end"].append(latest_availability)
        else:
            available_times["start"].append(earliest_avaialability)
            available_times["end"].append(latest_availability)

    
    ### Adding the beggining of the day and end of the day
    if add_entry: available_times['start'].insert(0, entry_time)
    if add_exit: available_times['end'].append(exit_time)
        
    ### Base check
    # for i, event in enumerate(events):
    #     start = event["start"].get("dateTime", event["start"].get("date"))
    #     end = event["end"].get("dateTime", event["end"].get("date"))
    #     print(f"Event {i} start: {start}")
    #     print(f"Event {i} end: {end}")



    result = f"Available times on {day} are:\n"
    event_duration_percentage = (event_duration / 60) * 100
    for start, end in zip(available_times['start'], available_times['end']):
        if start + event_duration_percentage > exit_time : break
        if start >= end: continue
        result += f"{start} to {end}\n"

    print(f"Iterated: {result}")
    return result


#### General availability checker tool
@tool(args_schema=TimeAvailability)
def check_availability(
    start: str,
    end: str,
    event_duration: int = None,
) -> str:
    """Check avaialable timings by referencing a Google calendar"""
    start = datetime.fromisoformat(start)
    end = datetime.fromisoformat(end)
    day_diff = end - start
    day_diff = day_diff.days
    day_delta = timedelta(days = 1)
    tmp_start = start
    tmp_end = start + day_delta
    result = ""
    for i in range(day_diff):
        result += check_day_availability(tmp_start, tmp_end, event_duration)
        result += '\n'
        tmp_start = tmp_end
        tmp_end = tmp_end + day_delta
    print(f"FINAL: {result}")
    return result


































##### Creating LLMs and agents
#### Router LLM
### Pydantic model
class Router(BaseModel):
    """Analyze the incoming query and route it according to its content"""

    reasoning: str = Field(
        description="Step-by-step reasoning behind the classification."
    )
        # Reasoning behind why LLM made the decision it chose
    classification: Literal["ignore", "respond", "notify"] = Field(
        description=""""\
        The classification of an incoming query:\
        'emailing_agent': queries related to the email APIs,\
        'calendar_agent': queries related to using the calendar API,\
        'memory_agent': queries related to saving or retriving memory items.
        """
    )


































##### Creating the communications MAS nodes
#### Creating the agent's state
class AgentState(TypedDict):
    messages: Annotated[List[AnyMessage], operator.add]
    input_query: str    
    metrics: Annotated[dict[str, dict[str,int]], operator.or_]

#### Creating node functionalities
### Router node
def router_node(state: AgentState, config: RunnableConfig) -> \
Command[
    Literal[
        'email_agent', 
        'calendar_agent',
        'memory_agent', 
        '__end__'
    ]
]: 
    ### Setup
    metrics = state['metrics']
    store = config['configurable']['store']
    goto = "__end__"
    update = {}

    ### Invoke agent

    ### Analyze metric usage

    ### Setup next graph traversal

    return Command(goto=goto, update=update)



### Email agent node
def email_node(state: AgentState, config: RunnableConfig)-> \
Command[
    Literal[
        "__end__"
    ]
]:
    ### Setup
    metrics = state['metrics']
    store = config['configurable']['store']
    goto = "__end__"
    update = {}

    ### Invoke agent

    ### Analyze metric usage

    ### Setup next graph traversal

    return Command(goto=goto, update=update)


### Calendar agent node
def calendar_node(state: AgentState, config: RunnableConfig)-> \
Command[
    Literal[
        "__end__"
    ]
]:
    ### Setup
    metrics = state['metrics']
    store = config['configurable']['store']
    goto = "__end__"
    update = {}

    ### Invoke agent

    ### Analyze metric usage

    ### Setup next graph traversal

    return Command(goto=goto, update=update)


### Memory agent node
def memory_node(state: AgentState, config: RunnableConfig)-> \
Command[
    Literal[
        "__end__"
    ]
]:
    ### Setup
    metrics = state['metrics']
    store = config['configurable']['store']
    goto = "__end__"
    update = {}

    ### Invoke agent

    ### Analyze metric usage

    ### Setup next graph traversal

    return Command(goto=goto, update=update)

























##### Assembling the communications MAS graph
communication_agent = StateGraph(AgentState)
communication_agent = communication_agent.add_node("router", router_node)
communication_agent = communication_agent.add_node("email_agent", email_node)
communication_agent = communication_agent.add_node("calendar_agent", calendar_node)
communication_agent = communication_agent.add_node("memory_agent", memory_node)
communication_agent = communication_agent.add_edge(START, "router")
communication_agent = communication_agent.compile()





















##### Visualize communications MAS
print(communication_agent.get_graph().draw_ascii())



















