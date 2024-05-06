from dotenv import load_dotenv
import logging
from typing import List

from langchain_community.document_loaders import WikipediaLoader
from langchain_community.graphs.neo4j_graph import Neo4jGraph
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from neo4j.exceptions import ClientError
from termcolor import colored

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

graph = Neo4jGraph()

# Embeddings & LLM models
embeddings = OpenAIEmbeddings()
embedding_dimension = 1536
llm = ChatOpenAI(model="gpt-4-turbo", temperature=0)

# Load from Wikipedia
documents = WikipediaLoader(query="One Piece", load_max_docs=5).load()

logger.info(colored(f"Loaded {len(documents)} documents", "green"))

# Split
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
child_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

# Ingest Parent-Child node pairs
parent_documents = parent_splitter.split_documents(documents)

for i, parent in enumerate(parent_documents):
    logger.info(colored(f"Processing parent {i}", "blue"))

    child_documents = child_splitter.split_documents([parent])
    params = {
        "parent_text": parent.page_content,
        "parent_id": i,
        "parent_embedding": embeddings.embed_query(parent.page_content),
        "children": [
            {
                "text": c.page_content,
                "id": f"{i}-{ic}",
                "embedding": embeddings.embed_query(c.page_content),
            }
            for ic, c in enumerate(child_documents)
        ],
    }
    # Ingest data
    graph.query(
        """
    MERGE (p:Parent {id: $parent_id})
    SET p.text = $parent_text
    WITH p
    CALL db.create.setVectorProperty(p, 'embedding', $parent_embedding)
    YIELD node
    WITH p 
    UNWIND $children AS child
    MERGE (c:Child {id: child.id})
    SET c.text = child.text
    MERGE (c)<-[:HAS_CHILD]-(p)
    WITH c, child
    CALL db.create.setVectorProperty(c, 'embedding', child.embedding)
    YIELD node
    RETURN count(*)
    """,
        params,
    )
    # Create vector index for child
    try:
        graph.query(
            "CALL db.index.vector.createNodeIndex('parent_document', "
            "'Child', 'embedding', $dimension, 'cosine')",
            {"dimension": embedding_dimension},
        )
    except ClientError:  # already exists
        pass
    # Create vector index for parents
    try:
        graph.query(
            "CALL db.index.vector.createNodeIndex('typical_rag', "
            "'Parent', 'embedding', $dimension, 'cosine')",
            {"dimension": embedding_dimension},
        )
    except ClientError:  # already exists
        pass


# INGEST HYPOTHETHICAL QUESTIONS
class Questions(BaseModel):
    """Generating hypothetical questions about text."""

    questions: List[str] = Field(
        ...,
        description=(
            "Generated hypothetical questions based on " "the information from the text"
        ),
    )


questions_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You are generating hypothetical questions based on the information "
                "found in the text. Make sure to provide full context in the generated "
                "questions."
            ),
        ),
        (
            "human",
            (
                "Use the given format to generate hypothetical questions from the "
                "following input: {input}"
            ),
        ),
    ]
)

question_chain = questions_prompt | llm.with_structured_output(Questions)

logger.info(colored("Generating hypothetical questions", "yellow"))

for i, parent in enumerate(parent_documents):
    logger.info(colored(f"Processing parent {i}", "blue"))
    questions = question_chain.invoke(parent.page_content).questions
    params = {
        "parent_id": i,
        "questions": [
            {"text": q, "id": f"{i}-{iq}", "embedding": embeddings.embed_query(q)}
            for iq, q in enumerate(questions)
            if q
        ],
    }
    graph.query(
        """
    MERGE (p:Parent {id: $parent_id})
    WITH p
    UNWIND $questions AS question
    CREATE (q:Question {id: question.id})
    SET q.text = question.text
    MERGE (q)<-[:HAS_QUESTION]-(p)
    WITH q, question
    CALL db.create.setVectorProperty(q, 'embedding', question.embedding)
    YIELD node
    RETURN count(*)
    """,
        params,
    )
    # Create vector index
    try:
        graph.query(
            "CALL db.index.vector.createNodeIndex('hypothetical_questions', "
            "'Question', 'embedding', $dimension, 'cosine')",
            {"dimension": embedding_dimension},
        )
    except ClientError:  # already exists
        pass

# INGEST SUMMARIES
summary_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You are generating concise and accurate summaries based on the "
                "information found in the text."
            ),
        ),
        (
            "human",
            ("Generate a summary of the following input: {question}\n" "Summary:"),
        ),
    ]
)

summary_chain = summary_prompt | llm

logger.info(colored("Generating summaries", "yellow"))
for i, parent in enumerate(parent_documents):
    logger.info(colored(f"Processing parent {i}", "blue"))
    summary = summary_chain.invoke({"question": parent.page_content}).content
    params = {
        "parent_id": i,
        "summary": summary,
        "embedding": embeddings.embed_query(summary),
    }
    graph.query(
        """
    MERGE (p:Parent {id: $parent_id})
    MERGE (p)-[:HAS_SUMMARY]->(s:Summary)
    SET s.text = $summary
    WITH s
    CALL db.create.setVectorProperty(s, 'embedding', $embedding)
    YIELD node
    RETURN count(*)
    """,
        params,
    )
    # Create vector index
    try:
        graph.query(
            "CALL db.index.vector.createNodeIndex('summary', "
            "'Summary', 'embedding', $dimension, 'cosine')",
            {"dimension": embedding_dimension},
        )
    except ClientError:  # already exists
        pass
