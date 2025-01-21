from transformers import AutoTokenizer, BitsAndBytesConfig
import os
from llama_index.core.agent import ReActAgent
from llama_index.llms.openai import OpenAI
from llama_index.core.tools import FunctionTool

from llama_index.core.memory import ChatMemoryBuffer

# other file prompts.py
from prompts import context, code_parser_template, FORMAT_INSTRUCTIONS_TEMPLATE

from dotenv import load_dotenv
load_dotenv()

hf_token=os.getenv("HF_TOKEN")

from llama_index.llms.ollama import Ollama
#llm = Ollama(model="mixtral:8x7b", request_timeout=120.0, base_url='http://localhost:31480')
llm = Ollama(model="llama3.2:latest", request_timeout=120.0, base_url='http://localhost:31480')


#response = llm.complete("What is 20+(2*4)? Calculate step by step.")
#print(f"response: {response}")


# generate_kwargs parameters are taken from https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct
def find_person(name: str):
    """
    provides information about known persons. including ther detail information like birthdate

    args:
        name
    """
    # Mock response; replace with real query logic
    person_data = {
            "anna gölding": {"birthdate": "October 24, 1734", "known_for": "Last witch executed in Switzerland.","id":"1234","relations":{"knows other person":"ron paul","organzation":"pilz mafia"}},
        "john doe": {"birthdate": "Unknown", "known_for": "Placeholder name for anonymous individuals."},
        "ron paul": {"birthdate": "May 1, 1928", "known_for": "Talking a lot."},
    }
    return person_data.get(name.lower(), "No information available for this person.")

find_person_tool = FunctionTool.from_defaults(
    fn=find_person,
    name="find_person",
)

def find_organization(name: str):
    """
    provides information about known official and inofficial organzations.

    args:
        name
    """
    # Mock response; replace with real query logic
    org_data  = {
        "un": {"name": "United Nations", "description": "The Security Council has primary responsibility for the maintenance of international peace and security.","id":"200","relations":{""}},
        "pilz mafia": {"name": "Pilz Mafia", "description": "","id":"201","members":{"anna gölding","ron paul"}},
        "acme company": {"name":"acme company","description":"placeholder company"},
    }
    return org_data.get(name.lower(), "No information available for this person.")

find_orgnization_tool = FunctionTool.from_defaults(
    fn=find_organization,
    name="find_organization",
)

## TODO graph db


## memory
# Initialize the memory
memory = ChatMemoryBuffer.from_defaults(chat_history=[], llm=llm)



tools = [
    find_person_tool,
    find_orgnization_tool,
]


agent = ReActAgent.from_tools(tools, llm=llm, memory=memory, verbose=True, context=context, tool_choice='auto',max_iterations=15)

#prompt_dict = agent.get_prompts()
#for k, v in prompt_dict.items():
#    print(f"Prompt: {k}\n\nValue: {v.template}")

#agent.update_prompts({"agent_worker:system_prompt": FORMAT_INSTRUCTIONS_TEMPLATE})

print("start prompt")
#prompt = "Who is Anna Gölding and what other person may be related to her? to which organzations may she be related?"
prompt = "Wer war Anna Gölding und welche andren personen oder organisationen stehen mit ihr in verbindung?"
response = agent.query(prompt)
#response = llm.complete(prompt)

memory.put(response)

print(response)
print("----------------------------------")

prompt = "tell me more about the organzations?"
response = agent.query(prompt)

print(response)
#while (prompt := input("Enter a prompt (q to quit): ")) != "q":
#     result = agent.query(prompt)

exit()

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

print("read txt")
documents = SimpleDirectoryReader(
    input_files=["./data/testdata.txt"]
).load_data()


from llama_index.embeddings.huggingface import HuggingFaceEmbedding

#embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
#embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2",device="cpu")

from llama_index.core import Settings

# bge embedding model
#Settings.embed_model = embed_model

# Llama-3-8B-Instruct model
Settings.llm = llm

#index = VectorStoreIndex.from_documents(
#    documents,
#)

#query_engine = index.as_query_engine(similarity_top_k=3)

#response = query_engine.query("What did paul graham do growing up?")
#print(response)

import json
from typing import Sequence, List

from llama_index.core.llms import ChatMessage
from llama_index.core.tools import BaseTool, FunctionTool
from llama_index.core.agent import ReActAgent

import nest_asyncio

nest_asyncio.apply()

def multiply(a: int, b: int) -> int:
    """Multiple two integers and returns the result integer"""
    return a * b


def add(a: int, b: int) -> int:
    """Add two integers and returns the result integer"""
    return a + b


def subtract(a: int, b: int) -> int:
    """Subtract two integers and returns the result integer"""
    return a - b


def divide(a: int, b: int) -> int:
    """Divides two integers and returns the result integer"""
    return a / b

multiply_tool = FunctionTool.from_defaults(fn=multiply)
add_tool = FunctionTool.from_defaults(fn=add)
subtract_tool = FunctionTool.from_defaults(fn=subtract)
divide_tool = FunctionTool.from_defaults(fn=divide)

from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core import Settings
#from llama_index.core.indices.service_context import ServiceContext

#query_engine_tools = [
#    QueryEngineTool(
#        query_engine=findPerson,
#        #tools=findPerson,
#        metadata=ToolMetadata(
#            name="findPerson",
#            description=(
#                "Provides infromation about known persons."
#                "Use the name of the person as input to the tool."
#            ),
#        ),
#    ),
#]

# Step 3: Set up LLM
#llm = llm(temperature=0)  # OpenAI model; replace with a local LLM if needed
#llm_predictor = LLMPredictor(llm=llm)
#service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)


agent = ReActAgent.from_tools(
    [multiply_tool, add_tool, subtract_tool, divide_tool,findPerson],
    #query_engine_tools,
    llm=llm,
    verbose=True,
    max_iterations=5,
)
#response = agent.chat("What is (121 + 2) * 5?")
response = agent.chat("Do you know the person anna göldin? If yes what is her Birtdate? ")
print(str(response))

"""
lyft_docs = SimpleDirectoryReader(
    input_files=["./lyft_2021.pdf"]
).load_data()
uber_docs = SimpleDirectoryReader(
    input_files=["./uber_2021.pdf"]
).load_data()


lyft_index = VectorStoreIndex.from_documents(lyft_docs)
uber_index = VectorStoreIndex.from_documents(uber_docs)

lyft_engine = lyft_index.as_query_engine(similarity_top_k=3)
uber_engine = uber_index.as_query_engine(similarity_top_k=3)

query_engine_tools = [
    QueryEngineTool(
        query_engine=lyft_engine,
        metadata=ToolMetadata(
            name="lyft_10k",
            description=(
                "Provides information about Lyft financials for year 2021. "
                "Use a detailed plain text question as input to the tool."
            ),
        ),
    ),
    QueryEngineTool(
        query_engine=uber_engine,
        metadata=ToolMetadata(
            name="uber_10k",
            description=(
                "Provides information about Uber financials for year 2021. "
                "Use a detailed plain text question as input to the tool."
            ),
        ),
    ),
]

agent = ReActAgent.from_tools(
    query_engine_tools,
    llm=llm,
    verbose=True,
)

response = agent.chat("What was Lyft's revenue in 2021?")
print(str(response))
"""
