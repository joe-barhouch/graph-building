from dotenv import load_dotenv

from langchain_community.graphs import Neo4jGraph

load_dotenv()

kg = Neo4jGraph(
    # url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD, database=NEO4J_DATABASE
)


print(kg.query("SHOW INDEXES"))

breakpoint()

print("###########################################")
print(kg.schema)
