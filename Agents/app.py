#from Agents import main_app
import streamlit as st
from main_app import process_app

#Dependency Keys
os.environ["api_key"] == st.secrets["FIREWORKS_API_KEY"]

# --- Define your backend function ---
def process_input(question : str) -> dict:
    response = process_app(question)
    return response

# --- Initialize session state ---
if "history" not in st.session_state:
    st.session_state.history = []
if "question" not in st.session_state:
    st.session_state.question = ""


# --- Build the Streamlit UI ---
st.title("Intelligent Model Selector Demo")

# Input fields
Question = st.text_input("Enter your Question:")
#task = st.selectbox("Choose a task:", ["Summarization", "Translation", "Question Answering"])



# Submit button
if st.button("Submit"):
    # Call the backend function with inputs
    result = process_input(Question)
    st.session_state.question = ""

    # Display the result
    st.success("Result from backend function:")
    #st.write(result)
    st.write(f"question: {Question}")
    st.write(f"Final Answer: {result['final_answer']}")
    st.write(f"Model Name: {result['modelname']}")
    st.write(f"Prompt Tokens: {result['Prompt_tokens']}")
    st.write(f"Completion Tokens: {result['Completion_tokens']}")
    st.write(f"Total Tokens: {result['Total_tokens']}")
    
    
    
