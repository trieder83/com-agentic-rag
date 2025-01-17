from transformers import AutoTokenizer
from dotenv import load_dotenv
load_dotenv()

hf_token=os.getenv("HF_TOKEN")

tokenizer = AutoTokenizer.from_pretrained(
    "meta-llama/Meta-Llama-3-8B-Instruct",
    token=hf_token,
)

stopping_ids = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>"),
]

# generate_kwargs parameters are taken from https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct

import torch
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.huggingface import HuggingFaceLLM

# Optional quantization to 4bit
# import torch
# from transformers import BitsAndBytesConfig

# quantization_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_compute_dtype=torch.float16,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_use_double_quant=True,
# )

llm = HuggingFaceLLM(
    #model_name="meta-llama/Meta-Llama-3-8B-Instruct",
    model_name="meta-llama/Llama-3.2-1B-Instruct",
    model_kwargs={
        "token": hf_token,
        "torch_dtype": torch.bfloat16,  # comment this line and uncomment below to use 4bit
        # "quantization_config": quantization_config
    },
    generate_kwargs={
        "do_sample": True,
        "temperature": 0.6,
        "top_p": 0.9,
    },
    #tokenizer_name="meta-llama/Meta-Llama-3-8B-Instruct",
    tokenizer_name="meta-llama/Llama-3.2-1B-Instruct",
    tokenizer_kwargs={"token": hf_token},
    stopping_ids=stopping_ids,
)

print("start prompt")
response = llm.complete("Who is Paul Graham?")

print(response)

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

print("read txt")
documents = SimpleDirectoryReader(
    input_files=["./data/testdata.txt"]
).load_data()


from llama_index.embeddings.huggingface import HuggingFaceEmbedding

embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

from llama_index.core import Settings

# bge embedding model
Settings.embed_model = embed_model

# Llama-3-8B-Instruct model
Settings.llm = llm

index = VectorStoreIndex.from_documents(
    documents,
)

query_engine = index.as_query_engine(similarity_top_k=3)

response = query_engine.query("What did paul graham do growing up?")

print(response)

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

agent = ReActAgent.from_tools(
    [multiply_tool, add_tool, subtract_tool, divide_tool],
    llm=llm,
    verbose=True,
)

response = agent.chat("What is (121 + 2) * 5?")
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
