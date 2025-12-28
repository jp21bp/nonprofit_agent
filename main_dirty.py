###### LAFF Agent
    # This project consists of an agent that will be used for LAFF
    # It will use sub-agents to complete a variety of tasks
    # The sub-agents will be found in other files

##### General setup
#### Importing libraries
### General libraries
import os, operator
from dotenv import load_dotenv
from pydantic import BaseModel, Field
### Manually created modules
from utilities import *
from prompts import *
### Typing libraries
from typing import Optional, Literal, List
from typing_extensions import TypedDict, Annotated
### Langchain/graph libraries
from langchain.agents import create_agent
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from langchain_core.messages import AnyMessage
### Model libraries
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_cohere import CohereEmbeddings
### Memory libraries
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.store.postgres import PostgresStore
from psycopg import Connection

#### Unpacking ENV variables
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")
cohere_api_key = os.getenv("COHERE_API_KEY")
postgres_conn = os.getenv("POSTGRES_CONN")

#### Setting up models
base_llm = ChatGoogleGenerativeAI(
    api_key = google_api_key,
    model = "gemini-2.5-flash-lite",
)
embedding_model = CohereEmbeddings(
    cohere_api_key=cohere_api_key,
    model = "embed-english-light-v3.0",
)

#### Setup DB storage utility
DB_NAME = "output.sqlite"
TABLE_NAME = "main_agent"
storage = Storage(DB_NAME, TABLE_NAME)

#### Creating memories
### Short-term memories
short_conn = sqlite3.connect(
    'checkpoints.sqlite',
    check_same_thread=False,
)
checkpointer = SqliteSaver(short_conn)

### Long-term memory
long_conn = Connection.connect(
    postgres_conn,
    autocommit=True
)
store = PostgresStore(
    long_conn,
    index={'embed': embedding_model, 'dims': 384}
)
store.setup()


#### Setting up genereal invocation configs
LG_USER_ID = "jp"
THREAD_NUM = 1

config = {
    'configurable':{
        'langgraph_user_id': LG_USER_ID,
        'thread_id': str(THREAD_NUM),
    }
}
























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
    items_list: list,
    namespace: str
):
    items = {}
    for item_name in items_list:
        to_store = store.get(
            namespace,
            item_name,
        )
        items[item_name] = to_store.value
    
    return items





























##### Prompt instructions
#### Creating the instructions
prompt_instructions = {
    "router_rules":{
        "grants": "Creating grant proposals, revising old proposals, critiquing propsals",
        "events": "Creating new events, status updates of events, modifying events",
        "emails": "Sending emails, reading emails, following up on email threads"
    }
}
#### Putting instructions into memory
namespace = (LG_USER_ID, "instructions")
for category, rule in prompt_instructions['router_rules'].items():
    store_items(
        store,
        namespace,
        f"{category}_rule",
        {"instruction": rule}
    )






















##### Creating LLMs
#### Router LLM
### Pydantic model
class Router(BaseModel):
    """\
    Analyze incoming queries and route them according to their
    contents and desired outcome.\
    """

    reasoning: str = Field(
        description="Step-by-step reasoning behind the classification."
    )
    classification: Literal["grants", "events", "emails",] = Field(
        description="""\
The classfication of an incoming query: \
'grants' for queries related to grant proposals, \
'events' for queries related to the management of events, \
'emails' for queries related to email tasks.\
"""
    )

### Creating model with pydantic output
llm_router = base_llm.with_structured_output(Router, include_raw=True)
























##### Creating node functionalities
#### Creating Agent State
class AgentState(TypedDict):
    messages: Annotated[List[AnyMessage], operator.add]
    metrics: Annotated[dict[str, dict[str, int]], operator.or_]
#### Creating router node
def main_router_node(state: AgentState) -> Command[
    # Literal["grants", "events", "emails", "__end__"]
    Literal["__end__"]
]:
    ## Setting up objects router LLM
    metrics = Metrics()
    update = {}
    goto = "__end__"

    ## Namespace to retrieve instructions
    instruct_namespace = (LG_USER_ID, "instructions")
    
    ## Retrieving instructions
    instructions = ["grants_rule", "events_rule", "emails_rule"]
    instructs = retrieve_items(store, instructions, instruct_namespace)

    ## Creating system prompt
    system_prompt = router_system_prompt.format(
        grants_route=instructs['grants_rule']['instruction'],
        events_route=instructs['events_rule']['instruction'],
        emails_route=instructs['emails_rule']['instruction'],
        examples=None
    )

    user_prompt = router_user_prompt.format(
        user_query = state['messages'][0].content
    )

    ## Invoking the router
    result = llm_router.invoke(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    )
    ai_msg = result['raw']
    message_update = {
        "messages": [ai_msg]
    }
    update = update | message_update

    ## Analyzing the invoked metrics
    extract = metrics.extract_tokens_used(ai_msg, "main_router_node")
    metrics = metrics.aggregate(extract)
    metrics_update = {
        "metrics": metrics.history
    }
    update = update | metrics_update

    ## Setting up next node to traverse
    # result = result['parsed']
    # if result.classification == 'grants':
    #     print('Classification: Grants')
    #     goto = 'grants'
    # elif result.classification == 'events':
    #     print('Classificaiton: Events')
    #     goto = 'events'
    # elif result.classication == 'emails':
    #     print('Classification: Emails')
    #     goto = 'emails'
    # else:
    #     raise ValueError(f"Invalid classificaiton: {result.classification}")
    
    ## Updating agent state
    return Command(goto=goto, update=update)




























##### Creating Agent
main_agent = StateGraph(AgentState)
main_agent = main_agent.add_node("main_router", main_router_node)
main_agent = main_agent.add_edge(START, "main_router")
main_agent = main_agent.compile(
    checkpointer=checkpointer,
    store=store
)























##### Visualize main agent grpah
print(main_agent.get_graph().draw_ascii())
























def main():
    print("DONE")


if __name__ == "__main__":
    main()
