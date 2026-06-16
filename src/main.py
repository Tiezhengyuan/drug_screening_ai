sys.path.append(".")
  from data_loader import load_prism_data
  from knowledge_base import build_knowledge_db
  from agent import build_agent

  def main():
      csv_dir = "./prism_data"
      pdf_dir = "./prism_papers"

      # 1. Ingest Data
      prism_data = load_prism_data(csv_dir)
      print("✓ Data Loaded")

      # 2. Build Knowledge DB
      kb_client = build_knowledge_db(pdf_dir)
      print("✓ Knowledge Base Initialized")

      # 3. Initialize Agent
      agent = build_agent(csv_dir, pdf_dir)

      # 4. Query Example
      response = agent.invoke({
          "input": "Compare MFI responses for compound 'PRISM_12345' vs DMSO control. "
                   "Also, summarize how PRISM handles pooling bias from recent literature."
      })
      print("\nAgent Output:\n", response["output"])

  if __name__ == "__main__":
      main()

