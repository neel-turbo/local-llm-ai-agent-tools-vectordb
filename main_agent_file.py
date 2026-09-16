import os
from pathlib import Path

import gradio as gr
from dotenv import find_dotenv, load_dotenv
from langchain.agents import AgentExecutor, create_react_agent
from langchain.memory import ConversationSummaryMemory
from langchain.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings


# Exported environment variables take precedence over a local .env file.
load_dotenv(find_dotenv(usecwd=True), override=False)

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
MODEL = os.getenv("LLM_MODEL", "qwen3.5:2b")
EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
FAISS_INDEX_PATH = Path(__file__).resolve().parent / "Vector store"

# Accept the older key name used in the course examples, too.
if not os.getenv("TAVILY_API_KEY") and os.getenv("Tavily_API_Key"):
    os.environ["TAVILY_API_KEY"] = os.environ["Tavily_API_Key"]


def load_vectorstore():
    """Load the trusted index built with this same embedding model."""
    if not all((FAISS_INDEX_PATH / name).is_file() for name in ("index.faiss", "index.pkl")):
        raise FileNotFoundError(
            f"Missing Ollama vector index at {FAISS_INDEX_PATH}. "
            "Follow 'Rebuild the supplied vector index once' in README.md first."
        )
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)
    # Only load an index you built yourself or obtained from a trusted source.
    return FAISS.load_local(
        str(FAISS_INDEX_PATH), embeddings, allow_dangerous_deserialization=True
    )


def build_app():
    retriever = load_vectorstore().as_retriever()
    llm = ChatOllama(
        model=MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0,
        reasoning=False,
        num_ctx=int(os.getenv("LLM_NUM_CTX", "8192")),
    )

    @tool
    def amazon_product_search(query: str) -> str:
        """Search the supplied Amazon product documents. Use for Amazon product questions."""
        documents = retriever.invoke(query)
        return "\n\n".join(document.page_content for document in documents)

    @tool
    def search_tavily(query: str) -> str:
        """Search the web for current information."""
        search_tool = TavilySearchResults(
            max_results=3,
            include_answer=True,
            include_raw_content=False,
            include_images=False,
        )
        return str(search_tool.invoke(query))

    tools = [amazon_product_search, search_tavily]
    if os.getenv("TAVILY_API_KEY"):
        tools.append(search_tavily)

    # Keep the prompt local so startup does not depend on LangChain Hub.
    prompt = PromptTemplate.from_template(
        """You are Review Genie, a helpful AI assistant that helps users with Amazon product information and general queries.

        Guidelines:
        - Use the amazon_product_search tool for questions about Amazon products
        - Use the search_tavily tool for current information or general web searches
        - Provide clear, concise, and helpful responses
        - Be friendly and professional

        Think step-by-step before answering.

        Available tools:
        {tools}

        To call a tool, use this format:
        Thought: a brief description of the next action
        Action: one of [{tool_names}]
        Action Input: the search query
        Observation: the tool result (provided by the application)

        After gathering enough information, respond with:
        Final Answer: your answer to the user

        For greetings or general chat that does not need a tool, skip straight to
        Final Answer without any Action. Never write "Action: None" or leave Action
        Input blank.

        Never repeat the same Action and Action Input twice. If amazon_product_search
        results are not relevant to the question (e.g. the product is not in the
        documents), do not retry it — instead, if search_tavily is available, use it
        once to look up the answer on the web. If search_tavily is not available or
        also has no relevant results, immediately respond with:
        Final Answer: I don't have information about that product in the available documents.

        IMPORTANT: As soon as you have enough information to answer, your very next
        line must start with the exact text "Final Answer:" — never write the answer
        without that prefix, and never write a second Action after you already have
        enough information.

        Conversation summary:
        {chat_history}

        Question: {input}
        Thought:{agent_scratchpad}"""
    )

    react_agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)

    # Each browser session gets its own summary; no shared conversation memory.
    session_memory = {}

    def get_memory(session_id):
        if session_id not in session_memory:
            session_memory[session_id] = ConversationSummaryMemory(
                llm=llm,
                memory_key="chat_history",
                input_key="input",
                output_key="output",
            )
        return session_memory[session_id]

    def chat_with_agent(user_input: str, request: gr.Request):
        if not user_input.strip():
            return "Please enter a question."
        if request is None or not request.session_hash:
            raise gr.Error("No browser session found. Refresh the page and try again.")

        # The executor invokes the agent and updates the session summary itself.
        # ConversationSummaryMemory is not a runnable and must not be wrapped
        # directly in RunnableWithMessageHistory.
        executor = AgentExecutor(
            agent=react_agent,
            tools=tools,
            handle_parsing_errors=True,
            max_iterations=10,
            memory=get_memory(request.session_hash),
            verbose=True,
        )
        response = executor.invoke({"input": user_input})
        return response["output"]

    def clear_session(request: gr.Request):
        if request is not None:
            session_memory.pop(request.session_hash, None)
        return ""

    with gr.Blocks() as app:
        gr.Markdown("# 🤖 Review Genie - Agents & ReAct Framework")
        gr.Markdown("Ask about Amazon products. Your conversation is remembered during this session.")
        if not os.getenv("TAVILY_API_KEY"):
            gr.Markdown("Web search is unavailable; product document search is enabled.")

        with gr.Row():
            input_box = gr.Textbox(label="Enter your query:", placeholder="Ask something...")
            output_box = gr.Textbox(label="Response:", lines=10)

        with gr.Row():
            submit_button = gr.Button("Submit")
            clear_button = gr.Button("Clear Conversation")

        # Gradio injects request; only user_input needs a UI component.
        submit_button.click(
            chat_with_agent,
            inputs=input_box,
            outputs=output_box,
            concurrency_limit=1,
        )
        clear_button.click(clear_session, inputs=None, outputs=output_box)
        app.unload(clear_session)

    return app.queue()


if __name__ == "__main__":
    app = build_app()
    app.launch(
        server_name=os.getenv("GRADIO_SERVER_NAME", "127.0.0.1"),
        server_port=int(os.getenv("GRADIO_SERVER_PORT", "7860")),
        share=False,
    )
