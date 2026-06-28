from openai import OpenAI
import fitz
import prompts 
import config
import os
client = OpenAI(
    base_url=os.environ["PROVIDER"],
    api_key=os.environ["GROQ_API_KEY"]
)
messages = [prompts.SYSTEM_PROMPT]
def extract_pdf_text(file_path) -> str:
    file_path = file_path.strip('"')
    text=""
    try:
        with fitz.open(file_path) as doc:
            text = "\f".join(page.get_text() for page in doc)
    except Exception as e:
        print(f"Error: {e}")
        return
    return text

def stream_response(messages):
    stream = client.chat.completions.create(
        model=config.MODEL_2,
        messages=messages,
        stream=True,
        temperature=config.TEMPERATURE,
        top_p=config.TOP_P
    )
    for chunk in stream:
        print(repr(chunk))
        token=chunk.choices[0].delta.content
        if token:
            yield token

def summarize_text(text):
    return {
        "role": "user",
        "content": f"Summarize:\n\n{text}"
    }

def chat(messages, user):
    messages.append({
        "role":"user",
        "content": user
    })

def clear_chat(messages):
    messages.clear()
    messages .append(prompts.SYSTEM_PROMPT)
    print("Chat Cleared")

def exit_chat():
    print("Chat Ended")

def main():
    print("Welcome to qwen, to summarize a document enter \"~\", \"clear\" to clear chat, \"exit\" to end chat")
    while True:
        user = input(">>")
        if user=="~":
            file_path=input("Paste the pdf path here: ")
            text=extract_pdf_text(file_path=file_path)
            if text is None:
                continue
            messages.append(summarize_text(text=text))
        elif user.lower()=="clear":
            clear_chat(messages=messages)
            continue
        elif user.lower()=="exit":
            exit_chat()
            break
        else:
            chat(messages=messages, user=user)
        ans = ""
        for token in stream_response(messages=messages):
            print(token, end="", flush=True)
            ans+=token
        print()
        messages.append({
            "role": "assistant",
            "content": ans
        })

if __name__ == "__main__":
    main()
