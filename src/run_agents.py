#!/usr/bin/env python3
"""Run the PRIM Drug Screening AI Agent System."""

from pathlib import Path
from prim_agent.config import DATA_DIR, PDF_DIR, INDEX_DIR
from prim_agent.rag_indexer import PRISMRAGIndexer
from prim_agent.rag_retriever import PRISMRAGRetriever
from prim_agent.agents import PRISMAgent

def main():
    print("=== PRIM Drug Screening AI Agent System ===")
    print("All components running locally.\n")
    
    # Step 1: Index PDF publications (run once, or if PDFs changed)
    print("Step 1: Indexing PRISM publications...")
    indexer = PRISMRAGIndexer()
    indexer.index_pdfs(PDF_DIR, INDEX_DIR)
    
    # Step 2: Initialize RAG retriever
    print("\nStep 2: Initializing RAG retriever...")
    retriever = PRISMRAGRetriever(INDEX_DIR)
    
    # Step 3: Initialize the main agent
    print("\nStep 3: Initializing PRISM Agent...")
    agent = PRISMAgent(retriever, DATA_DIR)
    
    # Step 4: Process a query
    query = "Identify compounds with strong anti-proliferative effects from PRIM data"
    print(f"\nStep 4: Processing query: '{query}'")
    
    results = agent.process_query(query)
    
    # Display results
    print("\n" + "="*60)
    print(results["summary"])
    print("="*60)
    
    # Display literature context
    print("\n📚 Relevant Literature Context:")
    for item in results.get("literature_context", []):
        print(f"\n Drug: {item['drug']}")
        for ctx in item.get("context", []):
            print(f"  Source: {ctx.get('source', 'unknown')}")
            print(f"  Content: {ctx.get('content', '')[:200]}...")
            print(f"  Relevance: {ctx.get('relevance_score', 0):.3f}")

if __name__ == "__main__":
    main()