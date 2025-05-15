# import os
# from fastapi import FastAPI
# from pydantic import BaseModel
# from dotenv import load_dotenv
# from langchain.chat_models import ChatOpenAI
# from langchain.prompts import PromptTemplate
# from langchain.chains import LLMChain
# from langchain_community.utilities import SearxSearchWrapper


# # Load env variables
# load_dotenv()

# # FastAPI app
# app = FastAPI()

# # Define your OpenAI model
# llm = ChatOpenAI(
#     openai_api_key=os.getenv("OPENAI_API_KEY"),
#     temperature=0.3,
#     model_name="gpt-3.5-turbo"
# )

# # SearxSearchWrapper configuration (use correct host/port)
# search = SearxSearchWrapper(searx_host="http://localhost:8080/")

# # Prompt template
# prompt_template = PromptTemplate(
#     input_variables=["search_result", "user_query"],
#     template="""
# This is the internet search:
# {search_result}

# This is the user query:
# {user_query}

# Answer the user query using the internet search.
# """,
# )

# # Request body schema
# class QueryInput(BaseModel):
#     query: str

# @app.post("/chat")
# def chat(query_input: QueryInput):
#     user_query = query_input.query

#     # Use LangChain's SearxSearchWrapper
#     try:
#         search_snippets = search.run(user_query)
#     except Exception as e:
#         search_snippets = f"Search failed: {e}"

#     # Format the prompt and run it through the LLM
#     try:
#         chain = LLMChain(llm=llm, prompt=prompt_template)
#         answer = chain.run({"search_result": search_snippets, "user_query": user_query})
#     except Exception as e:
#         answer = f"LLM error: {e}"

#     return {"response": answer}

# # Run with: python main.py
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)


import os
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.utilities import SearxSearchWrapper

# Load .env if needed
load_dotenv()

app = FastAPI()

# Initialize SearxNG wrapper
search = SearxSearchWrapper(searx_host="http://localhost:8080/")

# Prompt template
prompt_template = PromptTemplate(
    input_variables=["search_result", "user_query"],
    template="""
This is the internet search:
{search_result}

This is the user query:
{user_query}

Answer the user query using the internet search.
""",
)

# Input model
class QueryInput(BaseModel):
    chatbot_name: str  # OpenAI, Claude (only OpenAI supported here)
    action_type: str   # summary, internet search
    query: str
    api_key: str

@app.post("/chat")
def chat(query_input: QueryInput):
    chatbot = query_input.chatbot_name.lower()
    action = query_input.action_type.lower()
    query = query_input.query
    api_key = query_input.api_key

    # Check if chatbot is supported
    if chatbot != "openai":
        return {"error": f"Chatbot '{chatbot}' not supported. Only 'OpenAI' is available."}

    # Initialize OpenAI LLM with provided API key
    llm = ChatOpenAI(
        openai_api_key=api_key,
        temperature=0.3,
        model_name="gpt-3.5-turbo"
    )

    # Define LLM chain
    chain = LLMChain(llm=llm, prompt=prompt_template)

    # Action handling
    if action == "internet search":
        try:
            search_result = search.run(query)
        except Exception as e:
            search_result = f"Search failed: {e}"
        try:
            answer = chain.run({"search_result": search_result, "user_query": query})
        except Exception as e:
            answer = f"LLM error: {e}"

    elif action == "summary":
        # Just pass query directly as summary input
        try:
            summary_prompt = PromptTemplate.from_template("Summarize this:\n{input}")
            summary_chain = LLMChain(llm=llm, prompt=summary_prompt)
            answer = summary_chain.run({"input": query})
        except Exception as e:
            answer = f"LLM error: {e}"
    else:
        return {"error": f"Action '{action}' is not supported. Use 'summary' or 'internet search'."}

    return {"response": answer}

# Run with: python main.py
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
