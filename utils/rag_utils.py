from langchain_text_splitters import RecursiveCharacterTextSplitter

from openai import OpenAI

def legal_chunking(text, chunk_size=700, overlap=100):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=len,
    )
    return splitter.split_text(text)

def generate_answer(client, context, query):
    prompt = f"""
    You are a helpful educational tutor.

    CONTEXT:
    {context}

    QUERY: {query}
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful educational tutor."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        max_tokens=512,
    )
    return response.choices[0].message.content.strip()