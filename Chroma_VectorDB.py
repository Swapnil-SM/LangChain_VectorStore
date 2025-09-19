from langchain_chroma import Chroma
from dotenv import load_dotenv
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()

# Create LangChain documents for IPL players
doc1 = Document(
    page_content="Virat Kohli is one of the most successful and consistent batsmen in IPL history. Known for his aggressive batting style and fitness, he has led the Royal Challengers Bangalore in multiple seasons.",
    metadata={"team": "Royal Challengers Bangalore"}
)
doc2 = Document(
    page_content="Rohit Sharma is the most successful captain in IPL history, leading Mumbai Indians to five titles. He's known for his calm demeanor and ability to play big innings under pressure.",
    metadata={"team": "Mumbai Indians"}
)
doc3 = Document(
    page_content="MS Dhoni, famously known as Captain Cool, has led Chennai Super Kings to multiple IPL titles. His finishing skills, wicketkeeping, and leadership are legendary.",
    metadata={"team": "Chennai Super Kings"}
)
doc4 = Document(
    page_content="Jasprit Bumrah is considered one of the best fast bowlers in T20 cricket. Playing for Mumbai Indians, he is known for his yorkers and death-over expertise.",
    metadata={"team": "Mumbai Indians"}
)
doc5 = Document(
    page_content="Ravindra Jadeja is a dynamic all-rounder who contributes with both bat and ball. Representing Chennai Super Kings, his quick fielding and match-winning performances make him a key player.",
    metadata={"team": "Chennai Super Kings"}
)

# Combine documents into a list
docs = [doc1, doc2, doc3, doc4, doc5]

# ✅ Use free HuggingFace embeddings instead of Gemini/OpenAI (no API keys, no quota issues, no billing.)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# ✅ Create a Chroma vector store
vector_store = Chroma(
    embedding_function=embeddings,
    persist_directory='my_chroma_db',
    collection_name='sample'
)

# Add documents
vector_store.add_documents(docs)

# View stored docs (optional)
print(vector_store.get(include=['embeddings','documents','metadatas']))

# Search documents
ans = vector_store.similarity_search(
    query='Who among these are a bowler?',
    k=2  # number of similar documents to return
)
print("\n🔎 Similarity Search Results:")
for a in ans:
    print(a)

# Search with similarity score
ans_with_score = vector_store.similarity_search_with_score(
    query='Who among these are a bowler?',
    k=2
)
# print("\n Similarity Search with Scores:")
# for a, score in ans_with_score:
#     print(f"Doc: {a.page_content[:60]}... | Score: {score}")

# meta-data filtering
vector_store.similarity_search_with_score(
    query="",
    filter={"team": "Chennai Super Kings"}
)

# update documents
updated_doc1 = Document(
    page_content="Virat Kohli, the former captain of Royal Challengers Bangalore (RCB), is renowned for his aggressive leadership and consistent batting performances. He holds the record for the most runs in IPL history, including multiple centuries in a single season. Despite RCB not winning an IPL title under his captaincy, Kohli's passion and fitness set a benchmark for the league. His ability to chase targets and anchor innings has made him one of the most dependable players in T20 cricket.",
    metadata={"team": "Royal Challengers Bangalore"}
)

vector_store.update_document(document_id='1cc66ec2-1137-406d-9d9f-41de13593d9a', document=updated_doc1)

# delete document
vector_store.delete(ids=['1cc66ec2-1137-406d-9d9f-41de13593d9a'])