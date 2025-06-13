import datetime
import os
from openai import OpenAI

# --- Configuration ---
# Define the log file name at the top for clarity and to avoid NameErrors.
CONV_LOGFILE = "Conversations.md"

# It's best practice to handle the API key securely.
# The script already does this correctly by using environment variables.
try:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
except TypeError:
    print("Error: The OPENAI_API_KEY environment variable is not set.")
    exit()


# --- Function Definitions ---

def log_conversation(user_prompt, system_prompt, ai_response, log_file=CONV_LOGFILE):
    """
    Logs the entire conversation to a specified markdown file.
    This version was kept as it provides more detailed logging.
    """
    try:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        separator = f"\n{'='*60}\nConversation timestamp: {timestamp}\n{'='*60}\n"

        with open(log_file, "a", encoding="utf-8") as f:
            f.write(separator)
            # It can be helpful to log the system prompt for debugging purposes.
            # f.write("[System]:\n" + system_prompt + "\n\n")
            f.write("[User]:\n" + user_prompt + "\n\n")
            f.write("[AI]:\n" + ai_response + "\n\n")
            f.write("-" * 40 + "\n\n")
    except IOError as e:
        print(f"Error writing to log file: {e}")

def load_readme(directory):
    """
    Tries to find and read a README file in the given directory.
    """
    for name in ['README', 'README.md', 'README.txt']:
        path = os.path.join(directory, name)
        if os.path.isfile(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    return f.read()
            except IOError as e:
                print(f"Error reading README file at {path}: {e}")
                return None
    return None

def code_to_text(file_path):
    """
    Reads the content of a file and returns it as a string.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except IOError as e:
        print(f"Error reading file {file_path}: {e}")
        return ""

def recursive_codebase_build(root_directory, codebase=None):
    """
    Recursively walks through a directory and builds a dictionary
    of all Python file contents.
    """
    if codebase is None:
        codebase = {}

    for item in os.listdir(root_directory):
        full_path = os.path.join(root_directory, item)
        if os.path.isdir(full_path):
            recursive_codebase_build(full_path, codebase)
        elif item.endswith('.py'):
            # Use relative path for cleaner keys in the codebase dictionary.
            relative_path = os.path.relpath(full_path, start=os.getcwd())
            codebase[relative_path] = code_to_text(full_path)
    return codebase

def list_directory_tree(starting_directory):
    """
    Generates a string representation of the directory tree.
    """
    tree_string = ""
    for root, _, files in os.walk(starting_directory):
        # Calculate level for indentation
        level = root.replace(starting_directory, '').count(os.sep)
        indent = ' ' * 4 * level
        tree_string += f"{indent}{os.path.basename(root)}/\n"
        file_indent = ' ' * 4 * (level + 1)
        for f in files:
            tree_string += f"{file_indent}{f}\n"
    return tree_string

def build_system_prompt():
    """
    Constructs the system prompt by gathering context from the codebase.
    """
    cd = os.getcwd()
    readme_text = load_readme(cd)
    directory_structure = list_directory_tree(cd)
    codebase = recursive_codebase_build(cd)

    # Use a list to build the prompt for better readability and efficiency.
    prompt_parts = [
        "You are a helpful expert AI coding assistant. Your task is to provide concise, accurate, and helpful assistance for the user's coding problems based on the context provided.",
        "\n--- CONTEXT: README ---\n" + (readme_text if readme_text else "No README file found."),
        "\n--- CONTEXT: DIRECTORY STRUCTURE ---\n" + directory_structure,
        "\n--- CONTEXT: CODEBASE ---\n" + "\n\n".join(f"# File: {path}\n{code}" for path, code in codebase.items()),
        "\n--- END OF CONTEXT ---\n",
        "Please carry out the user's instructions carefully. Prioritize providing code solutions over prose."
    ]

    return "".join(prompt_parts)

# --- Main Execution ---

if __name__ == '__main__':
    print("Building context from your codebase...")
    system_prompt = build_system_prompt()
    print("Context built successfully. You can now ask questions.")

    while True:
        try:
            user_prompt = input("\nWhat is your question? (Type 'exit' to quit)\n> ")
            if user_prompt.strip().lower() == 'exit':
                print("Goodbye!")
                break

            response = client.chat.completions.create(
                # "gpt-4.1" is not a standard model name. Replaced with gpt-4o.
                # You can also use "gpt-4-turbo" or "gpt-3.5-turbo".
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            )

            ai_response = response.choices[0].message.content
            print("\n--- AI Response ---\n")
            print(ai_response)
            print("\n-------------------\n")

            # Log the conversation
            log_conversation(user_prompt, system_prompt, ai_response)

        except Exception as e:
            print(f"An error occurred: {e}")
            break
