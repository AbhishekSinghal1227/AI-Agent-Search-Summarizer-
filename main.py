from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

# for Prompt templet
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.output_parsers import OutputFixingParser, PydanticOutputParser


#Creating a AI Agent
from langchain.agents import create_tool_calling_agent, AgentExecutor   # AgentExecuter is basically  used to execute the AI Agent

#  For Search tools used
from tools import search_tool,wiki_tool,save_tool

import os
## LLM Setup

load_dotenv()

class ResearchResponse(BaseModel):

    topic: str
    summary: str    
    sources: list[str]
    tools_used: list[str]
    items: list[str] | None = None   # <--- NEW field for lists

api_key = os.environ.get("OPENAI_API_KEY")

llm = ChatOpenAI(model='gpt-3.5-turbo',openai_api_key=api_key)    # need to have a api key from openai
parser= PydanticOutputParser(pydantic_object= ResearchResponse)

# response = llm.invoke("What is the meaning of life?")
# print("\n🤖 AI Response:\n", response)

# Wrap parser with fixing parser to auto-correct schema mistakes
fixing_parser = OutputFixingParser.from_llm(parser=parser, llm=llm)

## Structure Output/Models or Prompt templet

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a research assistant. Use tools if needed to answer the user query.

            🚨 IMPORTANT:
            - Return ONLY valid JSON that matches the schema below.
            - Do NOT include schema definitions or extra text.
            - Fill in all fields with actual values.
            - Include sources where possible and provide a concise summary.
            - Always include the list of items separately
                • Provide a clear **summary** about the category (e.g., overview of U.S. National Parks).  
                • Include the **list of items** inside the `summary` field along with the explanation.  
                • Also ensure each item’s **name** is clearly mentioned in the list.
            {format_instructions}
            """,
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
    ).partial(format_instructions=parser.get_format_instructions())

# Creating and Running a Agent

tools =[search_tool,wiki_tool,save_tool]
agent = create_tool_calling_agent(
    llm= llm,
    prompt =prompt,
    tools=tools

)

agent_executor = AgentExecutor(agent=agent, tools=tools,verbose =True)  # verbose gives us rthe thaught process of the AIAgent if we don't want to see it we can make it False
query = input(
  "\n🤖 Hi, this is Abi — your Personal AI Agent!\n"
    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    "💡 What can I help you research today?\n\n👉 "
)

raw_response = agent_executor.invoke({"query": query})


try:
    structured_response = fixing_parser.parse(raw_response.get("output"))
    print("\n📌 Research Response:")
    print(f"Topic      : {structured_response.topic}")
    print(f"Summary    : {structured_response.summary}")
    print(f"Sources    : {', '.join(structured_response.sources)}")
    print(f"Tools Used : {', '.join(structured_response.tools_used)}")
   
except Exception as e:
    print("❌ Failed to parse the response:", e)
    print("Raw Response:", raw_response)


# UI USER INTERFACE
import gradio as gr
def run_agent_ui(query):
    raw_response = agent_executor.invoke({"query": query})
    try:
        structured_response = fixing_parser.parse(raw_response.get("output"))
        result = f"""
Topic      : {structured_response.topic}

Summary    : {structured_response.summary}

Sources    : {', '.join(structured_response.sources)}

"""
        return result
    except Exception as e:
        return f"❌ Failed to parse the response: {e}\nRaw Response: {raw_response}"

# ---- Gradio UI ----
with gr.Blocks() as demo:
    gr.Markdown("🤖 Abi — Your Personal AI Search Summariser")
    query_input = gr.Textbox(label="Enter your query:", placeholder="Ask me anything!", lines=2)
    submit_btn = gr.Button(" Submit ")
    output_box = gr.Textbox(label="AI Response", lines=30)

    submit_btn.click(fn=run_agent_ui, inputs=query_input, outputs=output_box)

demo.launch()




