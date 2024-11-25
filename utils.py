from groq import Groq
from dotenv import load_dotenv
import os


MODEL_NAME = "llama-3.1-70b-versatile"
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def get_response(prompt):
    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=1
    )
    return completion.choices[0].message.content


def get_prompt(query, chunks):
    prompt = f"""
            You are an expert at analyzing novels for high school English / Literature classes
            Given the query and the chunks retrieved from the novel, answer the query. 
            You can use some of your own knowledge if you recognize the novel.
            Here is the query: {query}\n
            Here are the chunks:\n"""
    for chunk in chunks:
        prompt += f"{chunk["text"]}\n"
    
    return prompt