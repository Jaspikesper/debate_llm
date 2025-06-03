import os
from openai import OpenAI
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)


codebase = {}
cd = os.getcwd()
contents = [f for f in os.listdir(cd)]
conversation = {}


system_prompt = "You are a helpful expert AI coding assistant optimized to provide personalized contextual assistance with the user's coding problems. When presented with a bug or coding problem, you provide concise, accurate and helpful coding assistance by identifying and addressing the issue. The user's codebase follows these instructions: \n"

def code_to_text(fname):
    text = open(fname).read()
    return text

def recursive_codebase_build(root_directory=os.getcwd()):
    print('root directory is: ' + root_directory)
    for fname in os.listdir(root_directory):
        if fname.endswith('.py'):
            codebase[fname] = code_to_text(os.path.join(root_directory, fname))
        if os.path.isdir(fname):
            recursive_codebase_build(os.path.join(root_directory, fname))


if __name__ == '__main__':
    recursive_codebase_build()
    system_prompt += str(codebase)
    system_prompt += "The user's codebase ends here. Carry out the user's instructions carefully. Do not provide unnecessary comments or prose, instead prioritize coding. If you fail at your coding task, you will certainly be abducted and killed."

    user_prompt = input("What is your question for ChatGPT? \n")

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )

    txt = response.choices[0].message.content
    print(txt)
