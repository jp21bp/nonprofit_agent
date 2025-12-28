###### RAG agent
    # Will be used on organization content blob found on main webpage
    # Recall: blob is in one big text file
        # Thus, we'll need a diff strat than Jopara RAG
            # Specifically, we won't need "document_loaders"
        # We'll simply open the file
        # Rest is similar to JoparaRAG
    # There is English and Spanish info, so we'll need Bilingual embed
        # Thus we'll also need an english and spanish chain similar to JoparaRAG
        # BUT, the info WON'T need to be returned in Spanish
            # ASSUMING the grants will be written in English
        # Thus, the spanish chain will be in charge of:
            # Translating initial english query into spanish
            # Finding the spanish related info
            # Turning the retrieved spanish info into enlgish
            # Return the english-translated info to user

##### General setup
#### Importing libraries
### General libraries
from dotenv import load_dotenv
import os
from pydantic import BaseModel, Field
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.runnables.base import RunnableLambda
from langchain_core.output_parsers.string import StrOutputParser
### Model libraries
from langchain_community.embeddings import JinaEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_cohere import ChatCohere
### Embedding libraries
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
### Utilities library
from utilities import *

#### Setting up environment
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")
jina_api_key = os.getenv("JINA_API_KEY")

#### Setting up models
### LLM model
    # Web: https://aistudio.google.com/welcome
llm = ChatGoogleGenerativeAI(
    api_key=google_api_key,
    model="gemini-2.5-flash",
)
### Embedding model
    # Web: https://jina.ai/
embedding_model = JinaEmbeddings(
    api_token=jina_api_key,
    model="jina-embeddings-v2-base-es"
)














##### File Embeddings
    # Note: we'll store all the embeddings in the same file
        # I.e., all diff. docs will be embedded into the same vstore
    # Thus, when loading the docs, we'll use "list.extend" over "list.append"
        # [1,2,3].append([4,5]) = [1,2,3,[4,5]]
        # [1,2,3].extend([4,5]) = [1,2,3,4,5]
#### Loading text file(s)
docs_path = os.path.join(os.getcwd(), 'links_and_documents')

def load_files(docs_path: str) -> list:
    loaded_docs = []
        # Since docs aren't that big, each doc will have its own space
            # I.e., 4 text files => "len(docs)" = 4
    for file in os.listdir(docs_path):
        if not file.endswith(".txt"): continue
            # Skips all non-text files
        file_path = os.path.join(docs_path, file)
        loader = TextLoader(file_path)
        loaded_docs.extend(loader.load())
    return loaded_docs
    
#### Split text into chunks
def split_docs(loaded_docs: list) -> list:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1105, chunk_overlap = 0,
    )
        #IMPORTANT NOTE:
            #1105 is needed to make "Chroma.from_docs()" line 99 work
                #Specifically, Jina embed model can only handle 512 chunked docs
            #I.e., if we use a smaller size, then len(split_docs) > 512
    chunked_docs = text_splitter.split_documents(loaded_docs)
    return chunked_docs


## Checking the size of the chunked docs for Jina embed
# sample = split_docs(load_files(docs_path))
# print(len(sample))

#### Creating vector store
directory = "IndexStore"
directory = os.path.join(docs_path, directory)
if not os.path.isdir(directory):
    # Create dir if not existant
    os.mkdir(directory)
if not os.listdir(directory): # Checks is dir is empty or not
    # This path is that dir is empty
    loaded_docs = load_files(docs_path)
        #Load the docs
    chunked_docs = split_docs(loaded_docs)
        #Split the docs
    vstore = Chroma.from_documents(
        chunked_docs, embedding_model,
        persist_directory=directory
    )
        # Store indexes in disk
else: 
    # This path means dir is not empty
    vstore = Chroma(
        persist_directory=directory, 
        embedding_function=embedding_model
    )
        # Load indexes from non-empty dir

#### Setting up retriever
retriever = vstore.as_retriever(
    search_type="similarity",
        #Literal["similarity", "mmr", "similarity_score_threshold"]
    search_kwargs={"k": 1}
        #Number of chunked documents to return
)
























##### Creating first chain - translating language
#### Setup prompt
SYST_PROMPT = """\
You are an experienced Senior translator between English and Spanish. \
Your role is to translate the input, English query into a Spanish verison. \
Your associated pydantic model will have the following attributes:
* original: stores the original English query
* translation: stored the translated Spanish query

Below are a few examples of the task:
<Examples>
Original: "What are the organizaton's documents associated with education projects?"
Translation: "Cuales son los documentos de la organization que estan associados con proyectos educacionales?"
</Examples>
"""
prompt = ChatPromptTemplate.from_messages([
    ("system", SYST_PROMPT),
    ("user", "This is the user's query: {query}")
])

#### Setup pydantic model
class Translator(BaseModel):
    """Will detect the original user query and translate it to spanish"""
    original: str = Field(...,
        description="User's query in English"
    )
    translation: str = Field(...,
        description="LLM's spanish translation of the user's query"
    )

#### Setup llm model
translator_model = llm.with_structured_output(Translator, include_raw=True)

#### Setup translator chain
translation_chain = prompt | translator_model




























##### Creating second chain - spanish processing
#### Create RunnableLambda
    #In order to have chain get the run similar search on translated query
inputs = RunnableLambda(
    lambda x: retriever.invoke(x['entrada'])
)
    # Returns a LIST of docs (see line 132)

#### Extracting the contents from the retrieved docs
extract = RunnableLambda(
    lambda x: [{"ingreso": doc.page_content} for doc in x]
    # lambda x: [doc.page_content for doc in x]
    # lambda x: [{"ingreso": doc.page_content} for doc in x]
)

#### Create prompt
SYST_PROMPT = """\
Eres un experto en traduciendo entre Espanol a Ingles. \
Tu rol es traduccir informacion del Espanol al Ingles. \
Traducce la informacion sin agregar contexto no encontrado \
en la informacion que se te de.
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYST_PROMPT),
    ("user", "{ingreso}")
    # ("user", "la informacion es: ")
])

#### Create model
modelo_espanol = llm

#### Creating chain
mini_cadena = prompt | modelo_espanol
cadena_espanol = inputs | extract | mini_cadena.map()
    # Returns a LIST of translated docs





















##### Main pipeline
#### Putting it all together
### Setting up Utilities class for working with data
DB_NAME = 'output.db'
TABLE_NAME = 'rag'
storage = Storage(DB_NAME, TABLE_NAME)

# utilities = Analyzer()
### Create the RAG class
class RAG():
    def __init__(self):
        self.metrics = Metrics()
        self.data_id = None

    def invoke(self, user_input: str, data_id: int = None, base_name: str = 'NONE') -> str:
        ### Updating data_id
        self.data_id = data_id

        ### First chain logic
        ## Invocation
        name = base_name + '_rag_first_chain'
        translation_results = translation_chain.invoke({
            "query": user_input
        })
        
        ## Save invocation to DB
        self.data_id = storage.save_data(translation_results, self.data_id, name)
        
        ## Analyze metrics
        extract = self.metrics.extract_tokens_used(translation_results['raw'], name)
        self.metrics = self.metrics.aggregate(extract)
        
        ## Get the pydantic results
        py_model = translation_results['parsed']
        original = py_model.original
        translation = py_model.translation



        ### Second chain
        ## Invocation
        context_spanish = cadena_espanol.invoke({
            "entrada": translation
        })
            #Returns list of strings
                # These strings are translated versions of the original doc

        ## Save and metric the results of the second chain
        spanish_context_string = []
        for i, item in enumerate(context_spanish):
            name = f'{base_name}_rag_second_chain_{i}'
            self.data_id = storage.save_data(item, self.data_id, name)
            extract = self.metrics.extract_tokens_used(item, name)
            self.metrics = self.metrics.aggregate(extract)
            spanish_context_string.append(item.content)



        ### Running similarity search on original query
        context_english = retriever.invoke(original)
        context_english = [x.page_content for x in context_english]

        ### Combining all info together
        final_context = "\n\n".join(context_english + spanish_context_string)
        # self.data_id = storage.save_data(final_context, self.data_id, 'final_rag_context')
            # ESta parte no es necesario
            # El node del main guarda este resultado
        return final_context, self.data_id, self.metrics






































