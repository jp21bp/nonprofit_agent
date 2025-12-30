###### LAFF Agent
    # This project consists of an agent that will be used for LAFF
    # It will use sub-agents to complete a variety of tasks
    # The sub-agents will be found in other files

##### Full documentation on subgraphs
    #https://docs.langchain.com/oss/python/langgraph/use-subgraphs#full-example-different-state-schemas

#### To read:
    # https://docs.langchain.com/oss/python/langchain/runtime
    # https://docs.langchain.com/oss/python/langchain/long-term-memory


##### General setup
#### Importing libraries
### General libraries
import os, operator
from dotenv import load_dotenv
from pydantic import BaseModel, Field
### Manually created modules
from utilities_module.utilities import *
from prompts import *
from grants_module.grants_agent import grants_agent
### Typing libraries
from typing import Optional, Literal, List
from typing_extensions import TypedDict, Annotated
### Langchain/graph libraries
from langchain.agents import create_agent
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from langchain_core.messages import AnyMessage,\
    SystemMessage, HumanMessage
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
# embedding_model = CohereEmbeddings(
#     cohere_api_key=cohere_api_key,
#     model = "embed-english-light-v3.0",
# )

#### Setup DB storage utility
DB_NAME = "output.sqlite"
TABLE_NAME = "main_agent"
storage = Storage(DB_NAME)

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
    # index={'embed': embedding_model, 'dims': 384}
)
store.setup()


#### Setting up genereal invocation configs
LG_USER_ID = "jp"
THREAD_NUM = 1

config = {
    'configurable':{
        'langgraph_user_id': LG_USER_ID,
        'thread_id': str(THREAD_NUM),
        'store': store,
        'storage': storage,
    }
}
    # Recall, the config is passed onto children subgraphs
























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





























##### Setting up LLMs
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







#### Grants Formatter LLM
    # This will be used to extract the theme of the request being sent
    # It will return a structured llm pydantic output
        # WHich will be extract and inserted into the grants subagent
### Pydantic model
class GrantsFormatter(BaseModel):
    """Extract general information related to a request about grant proposals"""
    
    theme: str = Field(
        description="The underlying theme related to the request"
    )

### Creating model with pydantic output
grants_formatter_llm = base_llm.with_structured_output(
    GrantsFormatter, 
    include_raw=True
)

### Few shot examples
## Data model/template
data_model = """\
Incoming query: {query}
> Theme: {theme}
"""
## Helper function
def grants_few_shots(examples):
    strs = ["Here are some examples to follow:"]
    for ex in examples:
        strs.append(
            data_model.format(
                query = ex["query"],
                theme = ex["theme"]
            )
        )
    return "\n--------------------------\n".join(strs)

## Creating examples
example1 = {
    "query" : "Write a proposal to help me fund an computer class training for highschool students",
    "theme": "High school education computer class project"
}

example2 = {
    "query": "Give me a proposal to teach young adults how to negotiate",
    "theme": "Adults business skill on negotiation"
}

example3 = {
    "query": "I need a proposal on strengthening community relationships",
    "theme": "Community building through relationships"
}

example4 = {
    "query": "A proposal on workshops that will teach financial literacy to community adults",
    "theme": "Financial literarcy project"
}

## Putting examples into long term memory
NUM_EXS = 4
examples = []
names = []
namespace = (LG_USER_ID, "few_shot_examples", "grants")
for i in range(NUM_EXS):
    exec(f"examples.append(example{i+1})")
    store_items(
        store,
        namespace,
        f"grants_{i}",
        examples[i]
    )
    exec(f"names.append('grants_{i}')")

# items = retrieve_items(store, namespace, names)
# few_shots = grants_few_shots(items.values())
# print(grants_formatter_system_prompt.format(examples=few_shots))




















































##### Creating node functionalities
#### Creating Agent State
class AgentState(TypedDict):
    messages: Annotated[List[AnyMessage], operator.add]
    metrics: Annotated[dict[str, dict[str, int]], operator.or_]
    grants_start_state: dict
    # events_start_state: dict
    # emails_start_state: dict

#### Creating router node
def main_router_node(state: AgentState, config: dict) -> \
Command[
    Literal[
        "grants_formatter", 
        "events_formatter", 
        "emails_formatter", 
        "__end__"
    ]
]:
    ## Setting up objects for router LLM
    metrics = Metrics()
    update = {}
    goto = "__end__"

    ## Namespace to retrieve instructions
    instruct_namespace = (LG_USER_ID, "instructions")
    
    ## Retrieving instructions
    instructions = ["grants_rule", "events_rule", "emails_rule"]
    instructs = retrieve_items(store, instruct_namespace, instructions)

    ## Creating system prompt
    system_prompt = router_system_prompt.format(
        grants_route=instructs['grants_rule']['instruction'],
        events_route=instructs['events_rule']['instruction'],
        emails_route=instructs['emails_rule']['instruction'],
        examples=None
    )

    ## Crating user prompt
    user_prompt = general_user_prompt.format(
        user_query = state['messages'][0].content
    )

    ## Invoking the router
    result = llm_router.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
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
    result = result['parsed']
    if result.classification == 'grants_formatter':
        print('Classification: Grants')

        # grants_start_state = {
        #     'messages': [user_query],
        #     'theme': 'educational projects',
        #     'doner_requirements': 'final proposal needs to be one page long',
        #     'num_revisions': 0,
        #     'max_revisions': 2,
        #     'id_counter': 1,
        # }
        goto = 'grants_formatter'
    elif result.classification == 'events_formatter':
        print('Classificaiton: Events')
        goto = 'events_formatter'
    elif result.classication == 'emails_formatter':
        print('Classification: Emails')
        goto = 'emails_formatter'
    else:
        raise ValueError(f"Invalid classificaiton: {result.classification}")
    
    ## Updating agent state
    return Command(goto=goto, update=update)













#### Formatter LLMs nodes
### Grants
def grants_formatter(state: AgentState, config: dict):
    ## Setting up objects for grants formatter
    metrics = Metrics()
    update = {}
    user_query = state['messages'][0].content

    ## Retrieving few shot examples
    namespace = (LG_USER_ID, "few_shot_examples", "grants")
    examples = retrieve_items(store, namespace, names)
    few_shots_exs = grants_few_shots(examples.values())

    ## Creating system prompt
    system_prompt = grants_formatter_system_prompt.format(
        examples = few_shots_exs
    )

    ## Creating user prompt
    user_prompt = general_user_prompt.format(
        query = user_query
    )

    ## Invoking grants formatter
    result = grants_formatter_llm.invoke([
        SystemMessage(content = system_prompt),
        HumanMessage(content = user_prompt),
    ])

    ## Extracting raw AI response
    ai_msg = result['raw']
    message_update = {
        "messages": [ai_msg]
    }
    update = update | message_update

    ## Analyzing the invoked metrics
    extract = metrics.extract_tokens_used(ai_msg, "grants_formatter_node")
    metrics = metrics.aggregate(extract)
    metrics_update = {
        "metrics": metrics.history
    }
    update = update | metrics_update

    ## Setting up the grants starting state
    grants_start_state = {
        'messages': [user_query],
        'metrics': state['metrics'],
        'theme': result['parsed'].theme,
        'doner_requirements': 'final proposal needs to be one page long',
        'num_revisions': 0,
        'max_revisions': 2,
        'id_counter': 0
    }
    grants_start_state_update = {
        "grants_start_state": grants_start_state,
    }

    update = update | grants_start_state_update

    ## Update state
    return update


### Events
def events_formatter(state: AgentState, config: dict):
    return
### Emails
def emails_formatter(state: AgentState, config: dict):
    return













#### Sub-agents
### Grants subagent 
def grants_agent_node(state: AgentState, config: dict):
    ### Setting up objects for the grants subagent
    metrics = Metrics()
    upgate = {}

    ## Invoking the grants agent
    result = grants_agent.invoke(
        state['grants_start_state'],
        config=config
    )
        # Will return the FINAL Grant's agent state values

    ## Extract the necessary attributes from Grant's final agent state
    update = {
        'messages': result['messages'], # Extra [] not needed bc already list
        'metrics' : result['metrics'],
    }

    ## Update main agent state
    return update

### Events subagent
def events_agent_node(state: AgentState, config: dict):
    # TODO: create an events agent
    return

### Emails subagent
def emails_agent_node(state: AgentState, config: dict):
    # TODO:  integrate email agent
    return

























##### Creating Agent
#### Initialize state graph
main_agent = StateGraph(AgentState)
#### Adding nodes
main_agent = main_agent.add_node("main_router", main_router_node)
main_agent = main_agent.add_node(grants_formatter)
main_agent = main_agent.add_node(events_formatter)
main_agent = main_agent.add_node(emails_formatter)
main_agent = main_agent.add_node("grants_subagent", grants_agent_node)
main_agent = main_agent.add_node("events_subagent", events_agent_node)
main_agent = main_agent.add_node("emails_subagent", emails_agent_node)
#### Adding edges
main_agent = main_agent.add_edge(START, "main_router")
main_agent = main_agent.add_edge("grants_formatter", "grants_subagent")
main_agent = main_agent.add_edge("events_formatter", "events_subagent")
main_agent = main_agent.add_edge("emails_formatter", "emails_subagent")
#### Compiling agent
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
