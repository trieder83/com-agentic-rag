from transformers import AutoTokenizer, BitsAndBytesConfig
import os
from dateutil import parser
from datetime import datetime
from functools import lru_cache
from llama_index.core.agent import ReActAgent
from llama_index.llms.openai import OpenAI
from llama_index.core.tools import FunctionTool
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, PromptTemplate
from llama_index.core.embeddings import resolve_embed_model
from llama_index.core.tools import QueryEngineTool
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.memory import ChatMemoryBuffer, ChatSummaryMemoryBuffer
from llama_index.extractors.entity import EntityExtractor
import tiktoken

#from langchain.chains.conversation.memory import ConversationBufferMemory

# Initialize memory
#memory = ConversationBufferMemory(memory_key="chat_history")


# other file prompts.py
from prompts import context, code_parser_template, FORMAT_INSTRUCTIONS_TEMPLATE, react_system_header_str

from dotenv import load_dotenv
load_dotenv()

hf_token=os.getenv("HF_TOKEN")
OLLAMA_URL=os.getenv("OLLAMA_URL")
LLM_MODEL=os.getenv("LLM_MODEL")
EMBEDDING_MODEL=os.getenv("EMBEDDING_MODEL")
#OLLAMA_URL='http://charon:31480'
#LLM_MODEL='llama3.2:latest'
#EMBEDDING_MODEL='snowflake-arctic-embed2'

from llama_index.llms.ollama import Ollama
#llm = Ollama(model="mixtral:8x7b", request_timeout=120.0, base_url='http://localhost:31480')
# linux 6g
llm = Ollama(model=LLM_MODEL, request_timeout=120.0, base_url=OLLAMA_URL, temperature=0)
# win 4g
#llm = Ollama(model="llama3.2:1b", request_timeout=300.0, base_url='http://localhost:11434', temperature=0)
# not tool support llm = Ollama(model="deepseek-r1:latest", request_timeout=300.0, base_url='http://localhost:11434', temperature=0)
#llm = Ollama(model="nemotron-mini:4b", request_timeout=300.0, base_url='http://localhost:11434', temperature=0)


#response = llm.complete("What is 20+(2*4)? Calculate step by step.")
#print(f"response: {response}")

# RAG index
print("read txt")
#documents = SimpleDirectoryReader("data").load_data()
documents = SimpleDirectoryReader(
    input_files=["/data/testdata.txt","/data/additionalinfo.txt"]
).load_data()

# bge-m3 embedding model
#Settings.embed_model = resolve_embed_model("local:BAAI/bge-small-en-v1.5")
Settings.embed_model = OllamaEmbedding(model_name=EMBEDDING_MODEL, base_url=OLLAMA_URL,embed_batch_size=100)
Settings.text_splitter = SentenceSplitter(chunk_size=512, chunk_overlap=30)
# ollama
Settings.llm = llm

index = VectorStoreIndex.from_documents(
    documents,
)

query_engine = index.as_query_engine(similarity_top_k=3)

knowledge_tool = QueryEngineTool.from_defaults(
    query_engine,
    name="knowledge_tool",
    description="""A RAG engine with some basic facts persons. Ask natural-language questions about persons and their properties and relations.
              if the knowledge_tool has no relatied information, ignore the answer.
              """
)


# generate_kwargs parameters are taken from https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct
@lru_cache(maxsize=10)
def find_person(name: str, **kwargs):
    """
    provides information about known persons. including ther detail information like birthdate

    args:
        name
    """
    # Mock response; replace with real query logic
    person_data = {
        "anna gölding": {"birthdate": "October 24, 1734", "known_for": "Last witch executed in Switzerland.","object_id":"1234","relations":{"knows other person":"ron paul","organzation":"pilz mafia"}},
        "john doe": {"birthdate": "Unknown", "known_for": "Placeholder name for anonymous individuals."},
        "ron paul": {"birthdate": "May 1, 1928", "known_for": "Talking a lot."},
        "miranda meyers": {"birthdate": "Aug 11, 1998", "known_for": "Miranda verkauft gerne verdorbens Eis. Das Eis erhält sie illegal von Litauen, wo es mit Mäusemilch hergestellt wird."},
    }
    return person_data.get(name.lower(), "No information available for this person.")

find_person_tool = FunctionTool.from_defaults(
    fn=find_person,
    name="find_person",
)

@lru_cache(maxsize=10)
def find_organization(name: str, **kwargs):
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
    return org_data.get(name.lower(), "No information available for this organization.")

find_orgnization_tool = FunctionTool.from_defaults(
    fn=find_organization,
    name="find_organization",
)


#def get_messages(name: str, min_daterange: datetime, max_daterange: datetime):
def get_messages(name: str, min_daterange: str, max_daterange: str, **kwargs):
    """
    Retrieve information about communications between two or more people within a given date range.

    # Example usage:
        name = "c1"
        min_daterange = ISO8601 date string
        max_daterange = ISO8601 date string
        messages = get_messages(name, min_daterange, max_daterange)

    Args:
        name (str): The name of the context always use c1.
        min_daterange (datetime): The start of the date range.
        max_daterange (datetime): The end of the date range.

    Returns:
        dict: A dictionary of messages for the given name within the date range.
    """
    # Mock response; replace with real query logic
    min_daterange_ts = parser.parse(min_daterange)
    max_daterange_ts = parser.parse(max_daterange)
    org_data = {
        "c1_1738446338": {"sender": "Ron Paul", "message": "Anna Gölding ist gestorben.", "timestamp": datetime(2025, 1, 1)},
        "c1_1738446338": {"sender": "Pilz Mafia", "message": "Hat Sie mit Boris Weed gesprochen oder ihn erwähnt? Sie wollte von ihm ein Sack voll Vogelfutter kaufen.", "timestamp": datetime(2025, 1, 5)},
    }

    result = {}
    for key, value in org_data.items():
        if key.startswith(name.lower()): # and min_daterange_ts <= value["timestamp"] <= max_daterange_ts:
            result[key] = value

    if not result:
        return "No information available for this timerange"

    return result

get_messages_tool = FunctionTool.from_defaults(
    fn=get_messages,
    name="get_messages",
)

## TODO graph db

# rag
# data/additionalinfo.txt

tools = [
    find_person_tool,
    find_orgnization_tool,
    get_messages_tool,
    knowledge_tool,
]

memory = ChatMemoryBuffer(token_limit=2000)


# extract only entities

entity_extractor = EntityExtractor(
    prediction_threshold=0.5,
    label_entities=False,  # include the entity label in the metadata (can be erroneous)
    device="cpu",  # set to "cuda" if you have a GPU
)
# ✅ Define an Entity Extractor for Organizations
org_extractor = EntityExtractor(
    entity_types=["ORGANIZATION"]  # Extracts only organizations
)

class OrganizationMemory(ChatMemoryBuffer):
    def chat(self, message):
        # Extract entities
        extracted_entities = org_extractor.extract(message)

        # Filter for only organizations
        organizations = [
            entity["text"] for entity in extracted_entities if entity["type"] == "ORGANIZATION"
        ]

        if organizations:
            filtered_message = "Organizations mentioned: " + ", ".join(organizations)
            super().chat(filtered_message)

org_memory = OrganizationMemory(token_limit=200)

agent = ReActAgent.from_tools(tools, llm=llm, verbose=True, context=context, tool_choice='auto',max_iterations=25, memory=memory, chat_history=org_memory) #, chat_history=memory)

# update agent system prompt
react_system_prompt = PromptTemplate(react_system_header_str)
agent.update_prompts({"agent_worker:system_prompt": react_system_prompt})
agent.reset()


#prompt_dict = agent.get_prompts()
#for k, v in prompt_dict.items():
#    print(f"Prompt: {k}\n\nValue: {v.template}")

#agent.update_prompts({"agent_worker:system_prompt": FORMAT_INSTRUCTIONS_TEMPLATE})

print("start prompt")
#prompt = "Who is Anna Gölding and what other person may be related to her? to which organzations may she be related?"
question_context = """context: c1, zeitbereich: 2025-02-01T00:00:00+00:00 to 2025-02-15T00:00:00+00:00 """
prompt = f"""Wer war Anna Gölding und welche andren personen oder organisationen stehen mit ihr in verbindung?
             Liste alle informationen und fakten die du findest in der antwort auf.
            questions context: {question_context}
            1. Analysiere die Person bekannt ist im find person tool.
            2. Analysiere die Person verbindungen zu anderen Personen oder Organisationen hat
            3. Prüfe ob Nachriten (messages) im context dieser Personen im gesuchten Zeitbereich statgefunden haben mit dem get_messages tool.
            4. Nenne die Anzahl der conversationen
            5. Nenne die Teilnehmer der conversationen
            6. Fasse den Inhalt der Kommunikation zusammen
            7. Prüfe ob Entitäten wie Personen, Organisationen oder Orte in der Nachrichten vorkommen, die bisher nicht bekannt sind.
            8. Kontrolliere ob alle Punkte dieser liste erfüllt sind
          """
response = agent.query(prompt)
#response = llm.complete(prompt)

#memory.save_context({"input": prompt}, {"output": str(response)})

print(f"AI: {response}")
org_memory.put(response)
print("----------------------------------")

#chat_history = [""]
chat_history = org_memory.get()
#chat_history =
#print (memory.load_memory_variables({})["chat_history"])
#prompt = f"there might be a conection. tell me more facts about the organzations mentioned? History: {chat_history}"
prompt = f"there might be a conection. tell the name of the organization. check if more facts about the organzation is avaliable? History: {chat_history}"
#memory.save_context({"input": prompt}, {"output": str(response)})
response = agent.query(prompt )

print(f"AI: {response}")
#while (prompt := input("Enter a prompt (q to quit): ")) != "q":
#     result = agent.query(prompt)

#memory.save_context({"input": prompt}, {"output": str(response)})

print("----------------------------------")
#chat_history = memory.load_memory_variables({})["chat_history"]
prompt = f"""provide the url to the person Anna Gölding in the format the fromat xx:/object:entity_type/id:object_id, the entity_type can be: PERSON, ORGANZATION.
the output must be a bullet list with where each item has this attributes: object, object type, url.
History: {chat_history}
"""

#memory.save_context({"input": prompt}, {"output": str(response)})
response = agent.query(prompt )

print(f"AI: {response}")

print("----------------------------------")
#chat_history = memory.load_memory_variables({})["chat_history"]
prompt = f"""print the relation between persons and organzations as ascii art. the persons MUST be in mentioned in the prompt or response.
Do not use any tools for this task.
The object should include the type and the name of the object. For example Person: <name of the person>.
History: {chat_history}
"""
response = agent.query(prompt )

print(f"AI: {response}")

