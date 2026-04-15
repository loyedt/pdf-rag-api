import chromadb

client = chromadb.PersistentClient(path="./chroma_data")
collection = client.get_or_create_collection("documents")

def add_docs(chunks, embeddings):
    for i, chunk in enumerate(chunks):
        collection.upsert(
            documents=[chunk],
            embeddings=[embeddings[i]],
            ids=[str(i)]
        )

def query_docs(query_embedding):
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3
    )
    return results["documents"][0]