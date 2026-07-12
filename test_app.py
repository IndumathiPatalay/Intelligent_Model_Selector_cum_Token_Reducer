from fireworks import Fireworks
import os
#from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
apikey=os.getenv("FIREWORKS_API_KEY")
client = Fireworks()

response = client.chat.completions.create(
    model="accounts/fireworks/models/kimi-k2p6",
    messages=[
        {
            "role": "system",
            "content": "You are a helpful assistant who would leverage the model given and predict the accurate answer with limited token usage"
        },
        {
            "role": "user",
            #"content": "If a train travels 60 km in 1.5 hours, what is its speed in km/h?"
            "content":"How do you determine who is the criminal in a law suite"
        }
    ]
)

answer = response.choices[0].message.content
usage_info = response.usage
Prompt_tokens = usage_info.prompt_tokens
Completion_tokens = usage_info.completion_tokens
Total_tokens = usage_info.total_tokens
model_name = "kimi-k2p6"

print("Model_name:",{model_name})
print("Final Answer:", {answer})
print("Prompt_Tokens:", {Prompt_tokens})
print("Completion_Tokens:", {Completion_tokens} )
print("Total_tokens:",{Total_tokens})
