from langchain.tools import tool
  from langchain.agents import create_react_agent, AgentExecutor
  from langchain_community.chat_models import ChatOllama
  from langchain.memory import ConversationBufferMemory
  from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
  import pandas as pd

  # Assume global loaders are initialized
  # from data_loader import load_prism_data
  # from knowledge_base import build_knowledge_db

  @tool
  def query_prism_data(metric: str, compound: str = None) -> str:
      """Return PRISM MFI/pooling metrics. `compound` optional."""
      # Integrate with actual DataFrame here
      return f"Requested metrics for {compound or 'all'}: [data snapshot placeholder]"

  @tool
  def retrieve_prism_knowledge(query: str) -> str:
      """Search PRISM publication knowledge base."""
      # Integrate with ChromaDB search
      return f"Knowledge retrieval for '{query}': [placeholder result]"

  def build_agent(csv_dir: str, pdf_dir: str):
      llm = ChatOllama(model="qwen2.5:14b", temperature=0.2)
      memory = ConversationBufferMemory(memory_key="chat_history")

      tools = [query_prism_data, retrieve_prism_knowledge]
      prompt = ChatPromptTemplate.from_messages([
          ("system", "You are a PRISM drug screening analyst. Use tools to inspect data and literature."),
          MessagesPlaceholder("chat_history"),
          ("human", "{input}"),
          MessagesPlaceholder("agent_scratchpad")
      ])

      agent = create_react_agent(llm, tools, prompt)
      executor = AgentExecutor(agent=agent, tools=tools, memory=memory, verbose=True)
      return executor
