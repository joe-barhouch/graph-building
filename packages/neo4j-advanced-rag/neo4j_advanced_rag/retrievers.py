import os

import openai
from dotenv import load_dotenv
from langchain_community.vectorstores.neo4j_vector import Neo4jVector
from langchain_openai import OpenAIEmbeddings

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
URL = os.getenv("NEO4J_URL")
USERNAME = os.getenv("NEO4J_USERNAME")
PASSWORD = os.getenv("NEO4J_PASSWORD")


openai.api_key = OPENAI_API_KEY


# TYPICAL RAG RETRIEVER
typical_rag = Neo4jVector.from_existing_index(
    OpenAIEmbeddings(api_key=OPENAI_API_KEY),
    index_name="typical_rag",
    url=URL,
    username=USERNAME,
    password=PASSWORD,
)

# PARENT RETRIEVER
parent_query = """
MATCH (node)<-[:HAS_CHILD]-(parent)
WITH parent, max(score) AS score // deduplicate parents
RETURN parent.text AS text, score, {} AS metadata LIMIT 1
"""

parent_vectorstore = Neo4jVector.from_existing_index(
    OpenAIEmbeddings(api_key=OPENAI_API_KEY),
    index_name="parent_document",
    retrieval_query=parent_query,
    url=URL,
    username=USERNAME,
    password=PASSWORD,
)

# HYPOTHETIC QUESTIONS RETRIEVER
hypothetic_question_query = """
MATCH (node)<-[:HAS_QUESTION]-(parent)
WITH parent, max(score) AS score // deduplicate parents
RETURN parent.text AS text, score, {} AS metadata
"""

hypothetic_question_vectorstore = Neo4jVector.from_existing_index(
    OpenAIEmbeddings(api_key=OPENAI_API_KEY),
    index_name="hypothetical_questions",
    retrieval_query=hypothetic_question_query,
    url=URL,
    username=USERNAME,
    password=PASSWORD,
)

# SUMMARY RETRIEVER
summary_query = """
MATCH (node)<-[:HAS_SUMMARY]-(parent)
WITH parent, max(score) AS score // deduplicate parents
RETURN parent.text AS text, score, {} AS metadata
"""

summary_vectorstore = Neo4jVector.from_existing_index(
    OpenAIEmbeddings(api_key=OPENAI_API_KEY),
    index_name="summary",
    retrieval_query=summary_query,
    url=URL,
    username=USERNAME,
    password=PASSWORD,
)
