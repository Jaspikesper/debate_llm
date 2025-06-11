import os
from openai import OpenAI

api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

codebase = {}
cd = os.getcwd()
contents = [f for f in os.listdir(cd)]
conversation = {}

def load_readme(directory):
    for name in ['README', 'README.md', 'README.txt']:
        path = os.path.join(directory, name)
        if os.path.isfile(path):
            with open(path, encoding='utf-8') as f:
                return f.read()
    return None

system_prompt = "You are a helpful expert AI coding assistant optimized to provide personalized contextual assistance with the user's coding problems. When presented with a bug or coding problem, you provide concise, accurate and helpful coding assistance by identifying and addressing the issue. The user's codebase follows these instructions: \n"

# Add README context if present
readme_text = load_readme(cd)
if readme_text:
    system_prompt = "README:\n" + readme_text + "\n\n" + system_prompt

def code_to_text(fname):
    text = open(fname, encoding='utf-8').read()
    return text

def recursive_codebase_build(root_directory=os.getcwd()):
    print('current file is directory is: ' + root_directory)
    for fname in os.listdir(root_directory):
        full_path = os.path.join(root_directory, fname)
        if fname.endswith('.py'):
            codebase[fname] = code_to_text(full_path)
        if os.path.isdir(full_path):
            recursive_codebase_build(full_path)

if __name__ == '__main__':
    recursive_codebase_build()
    system_prompt += str(codebase)
    system_prompt += "The user's codebase ends here. Carry out the user's instructions carefully. Do not provide unnecessary comments or prose, instead prioritize coding. If you fail at your coding task, you will certainly be abducted and killed."

    while True:
        user_prompt = input("What is your question for ChatGPT? \n")

        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )

        txt = response.choices[0].message.content
        print(txt)

        cont = input("\nDo you wish to continue the conversation? (yes to continue):\n")
        if cont.strip().lower() != "yes":
            print("Goodbye!")
            break
