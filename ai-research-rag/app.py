import streamlit as st
from sentence_transformers import SentenceTransformer
from endee import Endee

# --- 1. Page Configuration ---
st.set_page_config(page_title="AI Research RAG", page_icon="📚", layout="centered")

# --- 2. Load Resources (Cached for speed) ---
@st.cache_resource(show_spinner="Loading embedding model & connecting to Endee...")
def init_system():
    # Load the exact same model we used for ingestion
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Connect to local Endee server
    client = Endee(token="")
    client.set_base_url('http://localhost:8080/api/v1')
    index = client.get_index(name="research_papers")
    
    return model, index

model, index = init_system()

# --- 3. UI Layout ---
st.title("📚 AI Research RAG Assistant")
st.markdown("Powered by **Endee Vector Database** & Streamlit.")

# The search bar
query = st.text_input("Ask a question about recent AI research:", placeholder="e.g., What are the latest advancements in natural language processing?")

if query:
    st.divider()
    
    with st.spinner("Searching Endee database for relevant papers..."):
        # --- 4. The Vector Search ---
        # Convert the user's text question into a vector
        query_vector = model.encode([query])[0].tolist()
        
        # Search Endee for the top 3 closest matching vectors using the correct query() method
        results = index.query(
            vector=query_vector,
            top_k=3
        )

    # --- 5. The RAG Generation Prep ---
    st.subheader("🤖 RAG Context & Response")
    
    # Extract the metadata (title, abstract, etc.) from the results to build our LLM context
    context = ""
    for i, res in enumerate(results):
        # Endee usually returns metadata inside a 'meta' or 'metadata' key
        meta = res.get('meta', {})
        context += f"Paper {i+1}: {meta.get('title')}\nAbstract: {meta.get('abstract')}\n\n"

    # Simulated LLM Prompt Construction
    prompt = f"You are a helpful AI research assistant. Answer the user query based ONLY on the provided context. Cite the paper titles.\n\nContext:\n{context}\n\nQuery: {query}\n\nAnswer:"
    
    st.success("Successfully retrieved semantic matches from Endee! Here is the prompt that is constructed for the LLM:")
    with st.expander("View Raw RAG Prompt"):
        st.text(prompt)
        
    st.info("*To generate a real text response here, you would simply pass the prompt above into an LLM API like Groq, Gemini, or OpenAI.*")

    st.divider()

    # --- 6. Displaying the Semantic Search Results ---
    st.subheader("📄 Retrieved Source Papers")
    st.markdown("These are the top matches returned directly by Endee's vector search.")
    
    for res in results:
        meta = res.get('meta', {})
        with st.container():
            st.markdown(f"**[{meta.get('title', 'Unknown Title')}]({meta.get('url', '#')})**")
            st.caption(f"Authors: {meta.get('authors', 'Unknown')}")
            st.write(meta.get('abstract', 'No abstract available.'))
            st.markdown("---")