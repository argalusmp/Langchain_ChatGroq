import os
import getpass
import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from tavily import TavilyClient
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from fastapi.responses import StreamingResponse

# Ensure GROQ_API_KEY is set
if "GROQ_API_KEY" not in os.environ:
    os.environ["GROQ_API_KEY"] = getpass.getpass("Your GROQ_API_KEY: ")

# Set up Tavily client
tavily_api_key = "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"
tavily_client = TavilyClient(api_key=tavily_api_key)

# Inisialisasi FastAPI
app = FastAPI()

# Model Request/Response untuk API
class ChatRequest(BaseModel):
    question: str

class ChatResponse(BaseModel):
    reply: str

llm = ChatGroq(
    model="llama-3.2-3b-preview",
    temperature=0.7,
    max_tokens=900,  # Adjust based on response length needs
    timeout=10,
    max_retries=3,
)

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an AI assistant that provides up-to-date information from recent sources."),
        ("system", "You always provide responses that are clear, concise, and objective."),
        ("system", "If the user asks for an opinion, give a reasoned and thoughtful response."),
        ("user", "Here is some context: {context}\nQuestion: {question}")
    ]
)

chain = (
    PromptTemplate.from_template(
        """Given the user question below, classify it as either need take latest information from internet or not. Always answer using 'yes' or 'no'.

Do not respond with more than one word and answer with only lowercase.

<question>
{question}
</question>

Classification:""" 
    )
    | llm
    | StrOutputParser()
)

def ask_if_need_tavily(question):
    isYes = chain.invoke({"question": question})
    return isYes == "yes"

def stream_response(prompt, delay=0.01):
    """
    Streams response character by character with a natural typing effect.
    """
    try:
        response = llm.stream(prompt)
        for chunk in response:
            for char in chunk.content:
                yield char
                time.sleep(delay)  # Simulate typing delay
    except Exception as e:
        yield f"\n[Error in streaming response]: {str(e)}"

def fetch_latest_info_and_respond_stream(user_question):
    """
    Fetches latest info from Tavily and streams the response.
    """
    # Step 3a: Execute Tavily Search to get the latest information
    search_results = tavily_client.search(
        query=user_question,
        search_depth="advanced"
    ).get("results", [])

    # Combine results into a single context string
    context = "\n\n".join([result["content"] for result in search_results])

    # Step 3b: Prepare the prompt with context from Tavily
    prompt = prompt_template.format(context=context, question=user_question)

    # Step 3c: Stream the LLM response
    return stream_response(prompt)

# Endpoint FastAPI untuk Chatbot
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    user_input = request.question

    # Periksa apakah memerlukan Tavily
    need_internet = ask_if_need_tavily(user_input)

    if need_internet:
        # Gunakan Tavily untuk mendapatkan informasi terbaru
        search_results = tavily_client.search(
            query=user_input,
            search_depth="advanced"
        ).get("results", [])

        context = "\n\n".join([result["content"] for result in search_results])
        prompt = prompt_template.format(context=context, question=user_input)
    else:
        # Prompt biasa tanpa Tavily
        prompt = user_input

    # Streaming respons secara bertahap
    async def stream_generator():
        try:
            response = llm.stream(prompt)
            for chunk in response:
                for char in chunk.content:
                    yield char
        except Exception as e:
            yield f"[Error in streaming response]: {str(e)}"

    return StreamingResponse(stream_generator(), media_type="text/plain")
