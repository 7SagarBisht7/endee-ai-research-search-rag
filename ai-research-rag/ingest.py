import arxiv
from sentence_transformers import SentenceTransformer
from endee import Endee, Precision

# 1. Initialize the embedding model
print("Loading embedding model (all-MiniLM-L6-v2)...")
model = SentenceTransformer('all-MiniLM-L6-v2')
DIMENSION = 384 

# 2. Fetch specific NLP papers
def fetch_arxiv_papers(query="cat:cs.CL OR all:\"natural language processing\"", max_results=100):
    """Fetches recent NLP papers from ArXiv."""
    print(f"Fetching top {max_results} papers for query: '{query}'...")
    client = arxiv.Client()
    search = arxiv.Search(
        query=query, 
        max_results=max_results, 
        sort_by=arxiv.SortCriterion.SubmittedDate
    )
    
    papers = []
    for result in client.results(search):
        papers.append({
            "id": result.get_short_id(),
            "title": result.title,
            "authors": [author.name for author in result.authors],
            "abstract": result.summary.replace('\n', ' '),
            "url": result.pdf_url
        })
    return papers

def main():
    # 3. Connect to local Endee Server
    print("Connecting to Endee server...")
    client = Endee(token="") 
    client.set_base_url('http://localhost:8080/api/v1')
    
    index_name = "research_papers"
    
    # Safely create or connect to the index
    print(f"Checking Endee index '{index_name}'...")
    try:
        client.create_index(
            name=index_name,
            dimension=DIMENSION,
            space_type="cosine",
            M=16,
            ef_con=128,
            precision=Precision.INT8
        )
        print("Created new index.")
    except Exception as e:
        # Catch the specific Endee conflict error if the index is already there
        if "already exists" in str(e).lower() or "conflict" in str(e).lower():
            print(f"Index already exists! Appending new NLP papers to it...")
        else:
            raise e
    
    # Grab the index reference
    index = client.get_index(name=index_name)

    # 4. Fetch Data
    papers = fetch_arxiv_papers(max_results=100)
    print(f"Successfully fetched {len(papers)} papers. Generating embeddings...")
    
    # 5. Generate Embeddings
    abstracts = [paper["abstract"] for paper in papers]
    embeddings = model.encode(abstracts)
    
    # 6. Format and Upsert into Endee
    print("Pushing vectors and metadata to Endee...")
    vectors_to_upsert = []
    
    for i, paper in enumerate(papers):
        vectors_to_upsert.append({
            "id": paper["id"],
            "vector": embeddings[i].tolist(), 
            "meta": {
                "title": paper["title"],
                "authors": ", ".join(paper["authors"]),
                "url": paper["url"],
                "abstract": paper["abstract"]
            }
        })
        
    # Endee will append these 100 highly relevant papers to your existing index
    index.upsert(vectors_to_upsert)
    print("Ingestion complete! Your Endee database is now packed with NLP research.")

if __name__ == "__main__":
    main()