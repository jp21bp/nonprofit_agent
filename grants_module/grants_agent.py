###### Grants Agent file
    # Will contain all components of the grants agent
    # Will be used in the main graph, through importation

##### General setup
#### Importing libraries
### General libraries
import os, operator
from dotenv import load_dotenv
from pydantic import BaseModel, Field
### Manually created modules
from utilities import *
### Typing libraries
from typing import TypedDict, Annotated, List
### Langchain-graph libraries
from langgraph.graph import StateGraph, START, END



















