import chromadb
from chromadb.utils import embedding_functions
import argparse, sys

parser=argparse.ArgumentParser()

parser.add_argument("--rmindex", help="delete chromdb")
parser.add_argument("--query", help="query")

args=parser.parse_args()

print(f"Args: {args}\nCommand Line: {sys.argv}\nquery: {args.query}")
print(f"Dict format: {vars(args)}")

CHROMA_DATA_PATH = "data/"
EMBED_MODEL = "model/all-MiniLM-L6-v2"
COLLECTION_NAME = "demo_docs"

client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)


embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=EMBED_MODEL
)

if args.rmindex == "Y":
	print(f"delete collection {COLLECTION_NAME}")
	client.delete_collection(COLLECTION_NAME)

collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=embedding_func,
    metadata={"hnsw:space": "cosine"},
)

documents = [
    "The latest iPhone model comes with impressive features and a powerful camera.",
    "Exploring the beautiful beaches and vibrant culture of Bali is a dream for many travelers.",
    "Einstein's theory of relativity revolutionized our understanding of space and time.",
    "Traditional Italian pizza is famous for its thin crust, fresh ingredients, and wood-fired ovens.",
    "The American Revolution had a profound impact on the birth of the United States as a nation.",
    "Regular exercise and a balanced diet are essential for maintaining good physical health.",
    "Leonardo da Vinci's Mona Lisa is considered one of the most iconic paintings in art history.",
    "Climate change poses a significant threat to the planet's ecosystems and biodiversity.",
    "Startup companies often face challenges in securing funding and scaling their operations.",
    "Beethoven's Symphony No. 9 is celebrated for its powerful choral finale, 'Ode to Joy.'",
]

genres = [
    "technology",
    "travel",
    "science",
    "food",
    "history",
    "fitness",
    "art",
    "climate change",
    "business",
    "music",
]

collection.add(
    documents=documents,
    ids=[f"id{i}" for i in range(len(documents))],
    metadatas=[{"genre": g} for g in genres]
)


query_results = collection.query(
    #query_texts=["Find me some delicious food!"],
    query_texts=[args.query],
    n_results=1,
)

print(f"vector {query_results}")



from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load the Mistral-3B model and tokenizer from the local directory
model_path = "./model/mistral-3b/models--qualcomm--Mistral-3B"  # Replace with your local directory path

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Prepare context data and the prompt
context = ["pizza has tomato", "broccoli is green"]
prompt_str = "Please generate related movies to {movie_name}"
movie_name = "food"

context_str = "\n".join(context)
final_prompt = f"\n{prompt_str.format(movie_name=movie_name)}"

# Combine the response from the index with the context and final prompt
final_input = f"{context_str}\n{final_prompt}"

inputs = tokenizer(final_input, return_tensors="pt").to(device)

# Generate a response
output = model.generate(
    inputs.input_ids,
    max_length=100,
    num_return_sequences=1,
    temperature=0.7,  # Adjust for creativity
    top_p=0.9,  # Adjust for diverse outputs
)

# Decode the output
response = tokenizer.decode(output[0], skip_special_tokens=True)

# Print the response
print("Generated response:", response)

#-----------------
#from llama_index.core.query_pipeline import QueryPipeline
#from llama_index.core import PromptTemplate

# try chaining basic prompts
#prompt_str = "Please generate related movies to {movie_name}"
#prompt_tmpl = PromptTemplate(prompt_str)

#p = QueryPipeline(chain=[prompt_tmpl, llm], verbose=True)
