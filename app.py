import os
import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter


st.set_page_config(page_title="PERSOMAL AI", page_icon="🤖")
st.title("🤖 Persomal AI")
st.write("지식재산권의 이해 과제_AI 챗봇")

files = [
    "about_me.txt",
    "personality.txt",
    "style.txt",
    "chat1.txt",
    "chat2.txt",
    "chat3.txt",
    "chat4.txt"
]

@st.cache_resource
def build_vector_db():
    docs = []
    for f in files:
        loader = TextLoader(f, encoding="utf-8")
        docs.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )
    split_docs = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    db = FAISS.from_documents(split_docs, embeddings)
    return db

db = build_vector_db()
llm = ChatOpenAI(model="gpt-4o-mini")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("PERSOMAL AI에게 질문해보세요")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    result = db.similarity_search(user_input, k=3)
    context = "\n\n".join([r.page_content[:800] for r in result])

    prompt = f"""
너는 이정헌이다.
아래 참고 정보를 바탕으로 자연스럽고 답해라.
반말로 편하게 말하고, 친구처럼 대화해라.
참고 정보에 없는 내용은 과하게 지어내지 마라.

[참고 정보]
{context}

[질문]
{user_input}
"""

    answer = llm.invoke(prompt).content

    st.session_state.messages.append({"role": "assistant", "content": answer})

    with st.chat_message("assistant"):
        st.markdown(answer)