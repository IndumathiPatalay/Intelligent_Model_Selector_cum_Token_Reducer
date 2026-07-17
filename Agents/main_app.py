# Install dependencies
# pip install langchain langgraph openai
from fireworks import Fireworks
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import os
#from openai import OpenAI
from dotenv import load_dotenv


# 1. Define the shared state
class AgentState(dict):
    question: str
    token: str
    modelname :str
    final_answer : str
    error: str
    observation: str
    Prompt_tokens : int 
    Completion_tokens : int
    Total_tokens : int

#Define dictionary for models
evaluation_dict = {
    "Code & Development": {
        "Code generation, reasoning & agentic tasks": [
            "DeepSeek-V4-Pro",
            "Kimi K2.6",
            "GLM 5.1",
            "MiniMax M2.7"
        ]
    },
    "AI Applications": {
        "AI agents with tool use": [
            "Kimi K2.6",
            "DeepSeek-V4-Pro",
            "GLM 5.1",
            "MiniMax M2.7"
        ],
        "General reasoning & planning": [
            "DeepSeek-V4-Pro",
            "Kimi K2.6",
            "GLM 5.1",
        #    "GPT-OSS 120B(medium)"
        ],
        "Long context & summarization": [
            "DeepSeek-V4-Pro",
            "Kimi K2.6",
            #"Qwen3.6 Plus",
            "GLM 5.1",
            "DeepSeek-V4-Flash"
        ],
        "Fast extraction, classification & search": [
            "DeepSeek-V4-Flash",
            #"MiniMax M2.5",
            #"Kimi K2.5",
            #"Step 3.7 Flash",
            #"GPT-OSS 20B(small)"
        ]
    },
    "Vision & Multimodal": {
        "Vision & document understanding": [
            "Kimi K2.6",
            "Qwen3.6 Plus",
            #"Step 3.7 Flash",
            #"Gemma 4 31B(small)"
        ],
        "Audio & video understanding": [
            #"Qwen3 Omni 30B A3B Instruct",
            #"NVIDIA Nemotron 3 Nano Omni 30B A3B"
            "DeepSeek-V4-Pro" # Using it as no model is available as per Fireworks site recommendation.
        ]
    },
    "Search & Retrieval": {
        "Embeddings & reranking": [
            #"Qwen3 Embedding 8B",
            #"Qwen3 Reranker 8B"
            "DeepSeek-V4-Pro" # Using it as no other effective model is available as per Fireworks site recommendation. 
        ]
    }
}

load_dotenv()
apikey=os.getenv("FIREWORKS_API_KEY")
client = Fireworks()

# 2. Create reasoning nodes
def think_node(state: AgentState):
    #print("In thinking node")
    response = client.chat.completions.create(
    model="accounts/fireworks/models/kimi-k2p6",
    messages=[
        {
            "role": "system",
            "content": f"""
            You are an expert in selecting the best model for inference from the list of models
            given in the {evaluation_dict}.

            Based on the least tokens utilized, for a given question respond with an appropriate model.
            Strictly always respond in JSON with no extra characters.

            The JSON must have the following keys:
            - "model_name": Specify the recommended model 
            - "token_efficiency": a list of sources used
            - "confidence": a float between 0 and 1
            - "Justification": reasons for choosing the model
            - "token_used": number of tokens used for input
            """
        },
        {
            "role": "user",
            "content": state['question']
        }
    ]
    
)    # Now parse the response
    parsed_response = response.choices[0].message.content
    # If the model returns JSON, load it safely
    import json
    try:
        parsed_json = json.loads(parsed_response)
        state['modelname'] = parsed_json.get("model_name")
        state['token'] = parsed_json.get("token")
    except json.JSONDecodeError:
        state['modelname'] = None  # fallback if parsing fails
    #print(parsed_json)
    return state


#2. Validator
def validator_node(state: AgentState):
    try:
        
        modelname = state["modelname"]
        #print(f"In validator node with modelname :{modelname}")
        if not modelname:
            raise ValueError("modelname is missing in state")
        model = f"accounts/fireworks/models/{modelname}"
        
        response = client.chat.completions.create(
            model=model.lower(),
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant."
                },
                {
                    "role": "user",
                    "content": (
                        f"give the final concise answer."
                    )
                }
            ]
        )
        state["error"] = None
        state["observation"] = "Good observation"
    except Exception as e:
        # Catch any unexpected errors and log them in state
        state["error"] = str(e)
        state["observation"] = "An error occurred during validator_node execution."
        error = state["error"]
        #print(f"error in try of validator is {error}")
    return state


#2.0 Controller
def controller(state: AgentState):
    modelname = state.get("modelname")
    error = state.get("error")

    #print(f"In controller node with modelname: {modelname}, error: {error}")

    if error:   # if any error string is present
        return "think"   # loop back to reasoning
    return "answer"      # finish after answering
      # finish after answering


#2.1 Answer Node

def answer_node(state: AgentState):
    
    try:
        modelname = state["modelname"]
        if not modelname:
            raise ValueError("modelname is missing in state")

        # Dynamically form the model string
        model = f"accounts/fireworks/models/{modelname}"
        #print(f"In answer node with modelname : {modelname}")
        response = client.chat.completions.create(
            model=model.lower(),
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant who would leverage the model given and predict the accurate answer with limited token usage"
                },
                {
                    "role": "user",
                    "content": state['question']
                }
            ]
        )

        answer = response.choices[0].message.content
        usage_info = response.usage
        Prompt_tokens = usage_info.prompt_tokens
        Completion_tokens = usage_info.completion_tokens
        Total_tokens = usage_info.total_tokens
        state["final_answer"] = answer
        state["Prompt_tokens"] = Prompt_tokens
        state["Completion_tokens"] = Completion_tokens
        state["Total_tokens"] = Total_tokens 
        state["error"] = None  # clear error if successful

    except Exception as e:
        state["error"] = str(e)
        state["final_answer"] = "An error occurred during answer_node execution."

    return state

# 3. Build the LangGraph workflow
graph = StateGraph(AgentState)
graph.add_node("think", think_node)
graph.add_node("validate",validator_node)
graph.add_node("answer", answer_node)

graph.set_entry_point("think")
graph.add_edge("think", "validate")
graph.add_conditional_edges("validate",controller)
graph.add_edge("validate", END)


def process_app(Question: str) -> dict:
    app = graph.compile()
    result_state = app.invoke({"question": Question})   
    return result_state

#print("token", result["token"])
#print("model_name:", result["modelname"])
#print("Final Answer:", result["final_answer"])
#print("Prompt_Tokens:", result["Prompt_tokens"])
#print("Completion_Tokens:", result["Completion_tokens"])
print("Total_tokens:",result["Total_tokens"])
