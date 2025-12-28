###### Grants Agent file
    # Will contain all components of the grants agent
    # Will be used in the main graph, through importation

##### General setup
#### Importing libraries
### General libraries
from dotenv import load_dotenv
from typing import TypedDict, Annotated, List
import os, operator
### Langchain-graph libraries
from pydantic import BaseModel, Field
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import StateGraph, END
from langchain_core.messages import AnyMessage,\
    HumanMessage, AIMessage, SystemMessage
from langchain_core.messages.utils import merge_message_runs
from langchain_core.prompts.chat import ChatPromptTemplate
### Model libraries
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mistralai.chat_models import ChatMistralAI
### Other python files 
from rag import RAG
from mini_agents import MiniAgent, SystemPrompts
from utilities import *

#### Setting up environment
load_dotenv()
mistral_api_key = os.getenv("MISTRAL_API_KEY")

#### Initialize LLMs
llm = ChatMistralAI(
    api_key= mistral_api_key,
    model_name= 'mistral-small'
)


#### Initialize RAG
rag = RAG()

#### Setting up DB storage utility
DB_NAME = 'output.db'
TABLE_NAME = 'grants_agent'
storage = Storage(DB_NAME, TABLE_NAME)

#### Setup checkpointer for agent
conn = sqlite3.connect('checkpoints.sqlite', check_same_thread=False)
    #"check_same_thread = False" => enables multi-thread usage
memory = SqliteSaver(conn)



















##### Creating main agent prompts
PLAN_PROMPT = """\
You are an expert writer tasked with writing \
a high level outline of a grant proposal. The \
grant outline will contain eight sections: Cover \
letter, Executive summary, Statement of Need, \
Goals and objectives, Methods and strategies, \
Plan of evaluation, Budget information, and \
Organizational background. You will be given a \
Pydantic model with fields corresponding to each one \
of these sections, where you will place the section \
outline.  Take into consideration the doner \
requirements, which are: {requirements}.\
"""

DRAFT_PROMPT = """\
You are an experienced, senior grant writer. Your task \
is to unite previously created proposal components of the grant. The \
first half of the grant is:

<First half of grant>
{first_half}
</First half of grant>

The second half of the grant is:

<Second half of grant>
{second_half}
</Secon half of grant>

The doner requirements are: {requirements}\
"""

CRITIQUE_PROMPT = """\
You are a senior Grants officer reviewing a grant propsal from \
a non-profit organization for the following theme: {theme}. \
Your task is to generate critique and recommendations for the user's submission draft. \
Provide detailed recommendations, including requests for length, \
depth, style, etc.

<User Submission Draft>
{draft}
</User Submission Draft>\
"""

INVESTIGATION_PROMPT = """\
You are an experienced mentor in providing support for writing grant \
proposal for non-profit organizations. Your task is to create a simple \
RAG prompt that will look into the applicant's organizational documents. \
You goals is to retrieve more contextual documents that will help improve \
the current grant draft based on provided techniques.

<Current draft>
{draft}
</Current draft>\
"""


































##### Creating main agent
#### Creating pydantic models
### For planning node
class SectionOutlines(BaseModel):
    """Stores the outline for each section of the grant proposal"""
    cover_letter: str = Field(
        description='Cover letter section of proposal'
    )
    executive_summary: str = Field(
        description='Executive summary section of proposal'
    )
    statement_of_need: str = Field(
        description='Statement of need section of proposal'
    )
    goals_and_objective: str = Field(
        description='Goals and objectives section of proposal'
    )
    methods_and_strategies: str = Field(
        description='Methods and strategies section of proposal'
    )
    plan_of_evaluation: str = Field(
        description='Plan of evaluation section of proposal'
    )
    budget_information: str = Field(
        description='Budget information section of proposal'
    )
    organizational_background: str = Field(
        description='Organizaitonal background section of proposal'
    )


#### Create Agent state
class AgentState(TypedDict):
    ### Static fields
    ## Set by user on starting state
    theme: str
    doner_requirements: str
    num_revisions: int
    max_revisions: int
    ## Created once by planning node
    plan: SectionOutlines
    ### Dynamic fields
    ## Mini agents
    summarizer_sections: Annotated[dict[str, AIMessage], operator.or_]
    mini_sections_first_half: Annotated[dict[str, AIMessage], operator.or_]
    mini_sections_second_half: Annotated[dict[str, AIMessage], operator.or_]
    ## General
    id_counter: int 
    metrics: Annotated[dict[str, dict[str,int]], operator.or_]
    messages: Annotated[List[AnyMessage], operator.add]
    draft: Annotated[List[AIMessage], operator.add]
    critique: Annotated[List[AIMessage], operator.add]
    rag_context: Annotated[List[str], operator.add]


##### Creating agent graph
    # General node template:
        # Model invocation -> Storage saving -> Metrics analysis -> Update
class GrantsAgent:
    #### Constructor
    def __init__(self, llm, rag, mini_sys_prompts: dict):
        graph = StateGraph(AgentState)
        ### Creating nodes
        ## General nodes
        graph.add_node('rag', self.rag_node)
        graph.add_node('planner', self.plan_node)
        graph.add_node('draft', self.draft_node)
        graph.add_node('critique', self.critique_node)
        graph.add_node('investigation', self.investigation_node) 
        ## Mini section agents
        graph.add_node('cover_sec', self.cover_node)
        graph.add_node('executive_sec', self.executive_node)
        graph.add_node('need_sec', self.need_node)
        graph.add_node('goal_sec', self.goal_node)
        graph.add_node('methods_sec', self.methods_node)
        graph.add_node('eval_sec', self.eval_node)
        graph.add_node('budget_sec', self.budget_node)
        graph.add_node('background_sec', self.background_node)
        ## Mini summarizer agents
        graph.add_node('summarizer_one', self.summarizer_one)
        graph.add_node('summarizer_two', self.summarizer_two)        


        ### Adding edges
        ## General
        graph.set_entry_point('rag')
        graph.add_edge('rag', 'planner')
        ## Planner to mini section agents
        graph.add_edge('planner', 'cover_sec')
        graph.add_edge('planner', 'executive_sec')
        graph.add_edge('planner', 'need_sec')
        graph.add_edge('planner', 'goal_sec')
        graph.add_edge('planner', 'methods_sec')
        graph.add_edge('planner', 'eval_sec')
        graph.add_edge('planner', 'budget_sec')
        graph.add_edge('planner', 'background_sec')
        ## Section agents to summarizer agents
        graph.add_edge('cover_sec','summarizer_one')
        graph.add_edge('executive_sec','summarizer_one')
        graph.add_edge('need_sec','summarizer_one')
        graph.add_edge('goal_sec','summarizer_one')
        graph.add_edge('methods_sec','summarizer_two')
        graph.add_edge('eval_sec','summarizer_two')
        graph.add_edge('budget_sec','summarizer_two')
        graph.add_edge('background_sec','summarizer_two')
        ## Summarizer agents to draft
        graph.add_edge('summarizer_one', 'draft')
        graph.add_edge('summarizer_two', 'draft')
        ## Conditional edge
        graph.add_conditional_edges(
            'draft',
            self.should_continue,
            {END: END, 'critique': 'critique'}
        )
        ## Conditional loop edges
        graph.add_edge('critique', 'investigation')
        graph.add_edge('investigation', 'draft')


        ### Graph compilation
        self.graph = graph.compile(
            checkpointer=memory,
        )

        ### Graph attributes
        self.llm = llm
        self.rag = rag
        self.mini_sys_prompts = mini_sys_prompts
        self.metrics = Metrics()


    #### Conditional edge logic
    def should_continue(self, state: AgentState):
        if state['num_revisions'] >= state['max_revisions']:
            return END
        return 'critique'


    #### Rag node logic
    def rag_node(self, state: AgentState):
        print('Inside rag node\n\n')
        ### Model invocation
        name = 'rag_node'
        user_query = state['messages'][0]
        id_counter = state['id_counter']
        rag_result, id_counter, rag_metrics = self.rag.invoke(user_query, id_counter,name)
        print(f"'rag' db index at index: {id_counter}")
        print(rag_result)
        print('\n' + '=' * 50 + '\n')

        ### Storage saving
        id_counter = storage.save_data(rag_result, id_counter, name)

        ### Metrics analysis
        self.metrics.history = self.metrics.history | rag_metrics.history
        print('Rag metrics')
        print(self.metrics.history)
        print('\n' + '=' * 50 + '\n')

        ### Update AgentState
        return {"rag_context": [rag_result], 
                "id_counter": id_counter,
                "messages": [AIMessage(content='Initial RAG')],
                'metrics': self.metrics.history,}


    #### Plan node logic
    def plan_node(self, state: AgentState):
        print('Inside plan node\n\n')
        ### Model invocation
        ## Setup prompts
        name = 'plan_node'
        USER_PROMPT = """\
Write all section outlines for the following theme: {theme}. \
The organizational context regarding this proposal and theme is:
<Organizational Context>
{context}
</Organizational Context>\
"""
        prompts = ChatPromptTemplate.from_messages([
            ('system', PLAN_PROMPT),
            ("user", USER_PROMPT)
        ]) 
        ## Invoke model
        msgs = prompts.invoke({
            "requirements": state['doner_requirements'],
            "theme": state['theme'],
            "context": state['rag_context'][-1] #Es una lista
        })
        model = self.llm.with_structured_output(SectionOutlines, include_raw=True)
        plan = model.invoke(msgs)
        print(f"'Plan' DB data is with index: {state['id_counter']}")
        print(plan)
        print('\n' + '=' * 50 + '\n')

        ### Storage saving
        id_counter = state['id_counter']
        id_counter = storage.save_data(plan, id_counter, name)

        ### Metrics analysis
        extract = self.metrics.extract_tokens_used(plan['raw'], name)
        self.metrics = self.metrics.aggregate(extract)
        print('plan metrics')
        print(self.metrics.history)
        print('\n' + '=' * 50 + '\n')

        ### Update AgentState
        return {"plan" : plan['parsed'], 
                'id_counter' : id_counter,
                'messages': [plan['raw']],
                'metrics': self.metrics.history,}


    #### Template for mini section agents
    def mini_agent_template(self, state: AgentState, key: str):
        ### Model invocation
        ## Creating prompts
        replacements = {
            'plan': state['plan'].__getattribute__(key),
            'theme': state['theme'],
            'requirements': state['doner_requirements'],
        }
        sys_prompt = self.mini_sys_prompts[key]
        ## Creating agent
        agent = MiniAgent(sys_prompt, replacements)
        print(f'Prompt for {key}:\n{agent.system}')
        print('\n' + '=' * 50 + '\n')
        ## Invoke model
        response = agent()

        ### Storage saving
            # Will be done in "self.minis_to_db" due to concurrency

        ### Metrics analysis
        extract = self.metrics.extract_tokens_used(response, key)
        self.metrics = self.metrics.aggregate(extract)
        print(f'metrics for {key}')
        print(self.metrics.history)
        print('\n' + '=' * 50 + '\n')

        ### Update AgentState
        ## Formatting how AgentState will be updated
        section_annotation = AIMessage(content=key)
        response = merge_message_runs(
            [section_annotation, response], 
            chunk_separator = " -- "
        )[0]
        return response


    #### Creating mini section agents/nodes
    def cover_node(self, state: AgentState):
        key = 'cover_letter'
        response = self.mini_agent_template(state, key)
        return {'mini_sections_first_half': {'section_one': response}, 
                'messages': [response],
                'metrics': self.metrics.history,}

    def executive_node(self, state: AgentState):
        key = 'executive_summary'
        response = self.mini_agent_template(state, key)
        return {'mini_sections_first_half': {'section_two': response}, 
                'messages': [response],
                'metrics': self.metrics.history,}
    
    def need_node(self, state: AgentState):
        key = 'statement_of_need'
        response = self.mini_agent_template(state, key)
        return {'mini_sections_first_half': {'section_three': response}, 
                'messages': [response],
                'metrics': self.metrics.history,}

    def goal_node(self, state: AgentState):
        key = 'goals_and_objective'
        response = self.mini_agent_template(state, key)
        return {'mini_sections_first_half': {'section_four': response}, 
                'messages': [response],
                'metrics': self.metrics.history,}

    def methods_node(self, state: AgentState):
        key = 'methods_and_strategies'
        response = self.mini_agent_template(state, key)
        return {'mini_sections_second_half': {'section_one': response}, 
                'messages': [response],
                'metrics': self.metrics.history,}

    def eval_node(self, state: AgentState):
        key = 'plan_of_evaluation'
        response = self.mini_agent_template(state, key)
        return {'mini_sections_second_half': {'section_two': response}, 
                'messages': [response],
                'metrics': self.metrics.history,}

    def budget_node(self, state: AgentState):
        key = 'budget_information'
        response = self.mini_agent_template(state, key)
        return {'mini_sections_second_half': {'section_three': response}, 
                'messages': [response],
                'metrics': self.metrics.history,}

    def background_node(self, state: AgentState):
        key = 'organizational_background'
        response = self.mini_agent_template(state, key)
        return {'mini_sections_second_half': {'section_four': response}, 
                'messages': [response],
                'metrics': self.metrics.history,}


    #### Template for mini summarizer agents
    def summarizer_agent_template(self, state: AgentState, key: str):
        ### Model invocacion
        ## Creating prompts
        replacements = {
            'section_one': state[f'mini_sections_{key}']['section_one'].content,
            'section_two': state[f'mini_sections_{key}']['section_two'].content,
            'section_three': state[f'mini_sections_{key}']['section_three'].content,
            'section_four': state[f'mini_sections_{key}']['section_four'].content,
            'theme': state['theme'],
            'requirements': state['doner_requirements'],
        }
        sys_prompt = self.mini_sys_prompts[f'{key}']
        ## Creating agent
        agent = MiniAgent(sys_prompt, replacements)
        print(f'Prompt for {key}:\n{agent.system}')
        print('\n' + '=' * 50 + '\n')
        ## Invoking model
        response = agent()


        ### Storage saving
            # To avoid concurrency conflicts, this'll be done in
                    #"self.mini_to_db"
        
        ### Metrics analysis
        extract = self.metrics.extract_tokens_used(response, key)
        self.metrics = self.metrics.aggregate(extract)
        print(f'metrics for {key}')
        print(self.metrics.history) 
        print('\n' + '=' * 50 + '\n')

        ### Updating AgentState
        ## Formating the way the update will happen
        section_annotation = AIMessage(content = key)
        response = merge_message_runs(
            [section_annotation, response], 
            chunk_separator = " -- ",
        )[0]
        return response


    #### Creating mini summarizer agents/nodes
    def summarizer_one(self, state: AgentState):
        key = 'first_half'
        response = self.summarizer_agent_template(state, key)
        return {'summarizer_sections': {f'{key}': response}, 
                'messages': [response],
                'metrics': self.metrics.history,}
    
    def summarizer_two(self, state: AgentState):
        key = 'second_half'
        response = self.summarizer_agent_template(state, key)
        return {'summarizer_sections': {f'{key}': response}, 
                'messages': [response],
                'metrics': self.metrics.history,}



    #### Saving mini agents results to DB
    def minis_to_db(self, state: AgentState):
        id_counter = state['id_counter']
        to_save = [
            'mini_sections_first_half',
            'mini_sections_second_half',
            'summarizer_sections',
        ]
        for keys in to_save:
            for name, AiMsg in state[keys].items():
                id_counter = storage.save_data(AiMsg, id_counter, f'{keys}_{name}')
        return id_counter


    #### Loop nodes
    ### Draft node
    def draft_node(self, state: AgentState):
        print('Inside draft node\n\n')
        ### Saving previous mini agents
        id_counter = state['id_counter']
        if state['num_revisions'] == 0:
            print('Saving mini agents')
            id_counter = self.minis_to_db(state)
            print('\n' + '=' * 50 + '\n')

        ### Model invocation
        ## Creating system prompt
        name = f"draft_{state['num_revisions']}"
        replacements = {
            'first_half': state['summarizer_sections']['first_half'].content,
            'second_half': state['summarizer_sections']['second_half'].content,
            'requirements': state['doner_requirements'],
        }
        sys_prompt = DRAFT_PROMPT.format(**replacements)
        ## Joining all context lists
        all_context = ''
        for context in state['rag_context']:
            all_context += context
        ## Creating user prompt
        user_prompt = f"""\
Using the information above, create a full grant proposal draft for \
the following theme: {state['theme']}. Consider the following context, \
retrieved from the organization's webpage using RAG, when completing your task:

<RAG Organizational Context>
{all_context}
</RAG Organizational Context>\
"""
        ## Formatting prompts
        prompts = ChatPromptTemplate.from_messages([
            ('system', sys_prompt),
            ("user", user_prompt)
        ]) 
        msgs = prompts.invoke({})
        ## Invoking the model
        print('pre-invocation')
        draft = self.llm.invoke(msgs)
        print('post-invocation')

        ### Guardar los resultados
        id_counter = storage.save_data(draft, id_counter, name)

        ### Analizar los metricos
        extract = self.metrics.extract_tokens_used(draft, name)
        self.metrics = self.metrics.aggregate(extract)
        print(f'{name} metrics')
        print(self.metrics.history)
        print('\n' + '=' * 50 + '\n')

        ## Updating AgentState
        return {
            'messages': [draft],
            'draft': [draft], 
            'num_revisions': state['num_revisions'] + 1,
            'id_counter' : id_counter,
            'metrics': self.metrics.history,
        }


    ### Critique node
    def critique_node(self, state: AgentState):
        print('Inside critique node\n\n')
        ### Invocacion
        ## Creating prompts
        name = f"critique_{state['num_revisions']}"
        replacements = {
            'theme': state['theme'],
            'draft': state['draft'][-1].content,
        }
        sys_prompt = CRITIQUE_PROMPT.format(**replacements)
        prompts = ChatPromptTemplate.from_messages([
            ('system', sys_prompt),
            ("user", 'Execute your task.')
        ])
        msgs = prompts.invoke({})
        ## Invoking the model
        print('pre-invocation')
        critique = self.llm.invoke(msgs)
        print('post-invocation')

        ### Guardar los resultados
        id_counter = state['id_counter']
        id_counter = storage.save_data(critique, id_counter, name)

        ### Analizar los metricos
        extract = self.metrics.extract_tokens_used(critique, name)
        self.metrics = self.metrics.aggregate(extract)
        print(f'{name} metrics')
        print(self.metrics.history)
        print('\n' + '=' * 50 + '\n')

        ### Update AgentState
        return {
            'messages': [critique],
            'critique': [critique], 
            'id_counter': id_counter,
            'metrics': self.metrics.history,
        }


    ### Investigation node
    def investigation_node(self, state: AgentState):
        print('Inside investigation node\n\n')
        ### Primera invocacion
        ## Creating prompts
        base_name = f"investigation_{state['num_revisions']}"
        name = base_name + '_new_rag_prompt'
        replacements = {
            'draft': state['draft'][-1].content,
        }
        sys_prompt = INVESTIGATION_PROMPT.format(**replacements)
        user_prompt = f"""\
The draft above has been given the following critique: \
{state['critique'][-1].content}. Create the RAG prompt, which will be fed \
to the organizational's website RAG agent.\
"""
        prompts = ChatPromptTemplate.from_messages([
            ('system', sys_prompt),
            ("user", user_prompt)
        ])
        msgs = prompts.invoke({})
        ## Invoking the model
        print('pre-invocations-ONE')
        new_rag_prompt = self.llm.invoke(msgs)
        print('post-invocations-ONE')

        ### Guardar resultados
        id_counter = state['id_counter']
        id_counter = storage.save_data(new_rag_prompt, id_counter, name)

        ### Analizar los metricos
        extract = self.metrics.extract_tokens_used(new_rag_prompt, name)
        self.metrics = self.metrics.aggregate(extract)
        print(f'{name} metrics')
        print(self.metrics.history)
        print('\n' + '=' * 50 + '\n')



        ### Segunda invocacion
        name = base_name + '_new_rag_context'
        user_query = new_rag_prompt.content
        print('pre-invoke-TWO')
        rag_result, id_counter, rag_metrics = self.rag.invoke(user_query, id_counter, name)
        print('post-invoke-TWO')
        print(f"{name} db index at index: {id_counter}")
        print(rag_result)
        print('\n' + '=' * 50 + '\n')

        ### Guardar los resultados
        id_counter = storage.save_data(rag_result, id_counter, name)

        ### Analizar los metricos
        self.metrics.history = self.metrics.history | rag_metrics.history
        print(f'{name} metrics')
        print(self.metrics.history)
        print('\n' + '=' * 50 + '\n')

        ### Updat AgentState
        return {
            'messages': [new_rag_prompt, AIMessage(content=f'{name}')],
            'rag_context': [rag_result],
            'id_counter': id_counter,
            'metrics': self.metrics.history,
        }






















