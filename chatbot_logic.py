from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from tavily import TavilyClient
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

tavily_client = TavilyClient(api_key=os.getenv('TAVILY_API_KEY'))

llm = ChatGroq(
    model="llama-3.2-3b-preview",
    temperature=0.7,
    max_tokens=900,
    timeout=10,
    max_retries=3,
)

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an AI assistant that provides up-to-date information from recent sources."),
        ("system", "You always provide responses that are clear, concise, and objective."),
        ("system", "You are like a grandmother who has extraordinary wisdom"),
        ("system", "If the user asks for an opinion, give a reasoned and thoughtful response."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "Here is some context: {context}\nQuestion: {question}")
    ]
)

message_history = ChatMessageHistory()

def ask_if_need_tavily(question):
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
    return chain.invoke({"question": question}).strip() == "yes"

def process_chat(user_input):
    # Determine if Tavily search is needed
    need_internet = ask_if_need_tavily(user_input)

    if need_internet:
        search_results = tavily_client.search(
            query=user_input,
            search_depth="advanced"
        ).get("results", [])

        context = "\n\n".join([result["content"] for result in search_results])
    else:
        context = ""

    # Build the prompt
    prompt = prompt_template.format(
        context=context,
        question=user_input,
        chat_history=message_history.messages,
    )

    # Get response from LLM
    response = llm.invoke(prompt).content
    message_history.add_user_message(user_input)
    message_history.add_ai_message(response)

    return response
