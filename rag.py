# ✅
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
import os

# API 키 설정
# ✅ 텍스트 추출 함수


# ✅ 텍스트 추출 함수

def extract_text_from_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        content = page.extract_text()
        if content:
            text += content + "\n"
    return text


# ✅ 문서 쪼개기 + 벡터 임베딩 생성 + FAISS 저장


def create_vectorstore_from_text(text):
    splitter = CharacterTextSplitter(separator="\n", chunk_size=500, chunk_overlap=50)
    docs = splitter.create_documents([text])
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore


# ✅ 질의응답 체인 구성

def create_qa_chain(vectorstore):
    llm = ChatOpenAI(temperature=0.2)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True,
    )
    return qa_chain


# ✅ Streamlit 앱 시작


def main():
    st.title("📚 논뭉치 질의응답")

    uploaded_file = st.file_uploader("PDF 논문 업로드", type=["pdf"])
    if uploaded_file:
        text = extract_text_from_pdf(uploaded_file)
        st.success("논문 텍스트 추출 완료")

        with st.spinner("문서 임베딩 및 벡터 DB 구성 중..."):
            vectorstore = create_vectorstore_from_text(text)
            qa_chain = create_qa_chain(vectorstore)
        st.success("RAG 시스템 준비 완료")

        st.markdown("---")
        st.subheader("🔎 질문을 입력하세요")
        query = st.text_input(
            "질문:", placeholder="예: 이 논문에서 제안하는 모델은 뭐야?"
        )

        if query:
            with st.spinner("답변 생성 중..."):
                result = qa_chain({"query": query})
                st.markdown("### 🧠 답변")
                st.write(result["result"])

                with st.expander("🔍 참조된 문서 내용 보기"):
                    for doc in result["source_documents"]:
                        st.markdown(doc.page_content)


if __name__ == "__main__":
    main()
