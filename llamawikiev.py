# Import necessary modules
from llama_index.readers.web import SimpleWebPageReader
from llama_index import VectorStoreIndex, ServiceContext
from llama_index.vector_stores import DeepLakeVectorStore
from llama_index.llms import OpenAI
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from ragas.llama_index import evaluate
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv("llamawiki.env")
activeloop_token = os.getenv('ACTIVELOOP_TOKEN') 
openai_key = os.getenv('OPENAI_API_KEY')
activeloop_org_id = os.getenv('ACTIVELOOP_ORG_ID')
activeloop_dataset_name = os.getenv('ACTIVELOOP_DATASET_NAME')

# Load documents from a web page
reader = SimpleWebPageReader(html_to_text=True)
documents = reader.load_data(["https://en.wikipedia.org/wiki/Tesla_Cybertruck"])

# Configure vector store and LLLM
dataset_path = f"hub://{activeloop_org_id}/{activeloop_dataset_name}"
vector_store = DeepLakeVectorStore(dataset_path=dataset_path)
llm = OpenAI(openai_key)
service_context = ServiceContext.from_defaults(llm=llm, chunk_size=500, chunk_overlap=40)

# Create vector index
vector_index = VectorStoreIndex.from_documents(documents, service_context=service_context)

# Create query engine and query for information
query_engine = vector_index.as_query_engine()
response = query_engine.query("What are the key features of the Tesla Cybertruck?")
print(response)

# Define evaluation metrics
metrics = [faithfulness, answer_relevancy, context_precision, context_recall]

# Prepare evaluation questions and answers
eval_questions = [
    "How many reservations did Tesla receive within 1.5 days of the Cybertruck unveiling?",
    "According to Musk, what inspired the design of the Cybertruck?",
    "What is the overall battery pack weight for the Cybertruck?",
]
eval_answers = [
    "145,000",  # Incorrect answer
    "Blade Runner and Lotus Esprit",  # Correct answer
    "750kgs",  # Correct answer
]

# Evaluate the system's performance
results = evaluate(query_engine, metrics, eval_questions, eval_answers)
print(results)

#Output 
{'faithfulness': 0.8000, 
 'answer_relevancy': 0.7634, 
 'context_precision': 0.6000, 
 'context_recall': 0.8667}
