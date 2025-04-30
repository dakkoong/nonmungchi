import streamlit as st
import requests
import nltk
import xml.etree.ElementTree as ET
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

nltk.download("wordnet")

# OpenAI LLM 설정
llm = ChatOpenAI(
    model_name="gpt-4o",
    temperature=0.3,
    openai_api_key=""
)

lemmatizer = WordNetLemmatizer()

# PDF 텍스트 추출

def extract_text_from_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

# TF-IDF 키워드 추출

def extract_keywords_tfidf(text, top_k=10):
    vectorizer = TfidfVectorizer(stop_words="english", max_features=top_k * 2)
    X = vectorizer.fit_transform([text])
    keywords = vectorizer.get_feature_names_out()

    cleaned = set()
    for word in keywords:
        lemma = lemmatizer.lemmatize(word.lower())
        cleaned.add(lemma)

    cleaned = list(cleaned)
    while len(cleaned) < top_k:
        cleaned.append("AI")

    return cleaned[:top_k]

# arXiv API로 논문 검색

def search_arxiv(keywords, max_results=5):
    if not keywords:
        keywords = ["AI", "machine learning"]

    query = "+OR+".join([f"all:{k}" for k in keywords])
    url = f"http://export.arxiv.org/api/query?search_query={query}&start=0&max_results={max_results}"

    response = requests.get(url)
    if response.status_code != 200:
        st.error(f"arXiv API 호출 실패! 오류코드: {response.status_code}")
        return []

    root = ET.fromstring(response.content)
    papers = []
    for entry in root.findall("{http://www.w3.org/2005/Atom}entry"):
        title = entry.find("{http://www.w3.org/2005/Atom}title").text.strip()
        link = entry.find("{http://www.w3.org/2005/Atom}id").text.strip()
        abstract = entry.find("{http://www.w3.org/2005/Atom}summary").text.strip()
        papers.append({"title": title, "url": link, "abstract": abstract})
    return papers

# PromptTemplates
summary_prompt = PromptTemplate(
    input_variables=["content"],
    template="""
다음은 영어로 된 논문의 전체 텍스트입니다.

당신의 작업 목표는 이 논문을 대학원생이나 박사 연구자가 연구에 직접 활용할 수 있도록 자연스럽고 튜터처럼 친절하게 한국어로 요약하는 것입니다.

요약 방식은 다음을 지켜 주세요:
1. 논문 전체를 큰 챕터 단위로 구분해 주세요.
2. 각 챕터마다 핵심 주장, 새로운 개념/모델, 중요성, 실험 결과를 설명해 주세요.
3. '반드시 한국어'로 요약해 주세요.
4. 딱딱한 요약체를 피하고 튜터처럼 친절하게!
5. 주요 개념은 간단한 정의와 함께!
6. 논문 문제의식, 기존 연구와의 차이, 한계점, 미래 방향을 마지막에 정리해 주세요.
7. 안녕하세요 와 같은 인삿말은 빼주세요.
8. 답변에 **굵은 글씨(Bold)** (예: **텍스트**)를 '사용하지 말아주세요'.
9. 모든 내용은 일반 서술형 텍스트로 작성해 주세요.
10. Markdown 스타일 포맷팅(굵게, 이탤릭 등)은 사용하지 말아주세요.

[논문 전문]

{content}
"""
)

review_prompt = PromptTemplate(
    input_variables=["content"],
    template="""
다음은 영어로 된 논문의 전체 텍스트입니다.

당신의 작업 목표는 '이 논문에 대한 심층적인 리뷰(Review)'를 작성하는 것입니다.  
단순 요약이 아니라, 논문을 깊이 읽고 분석하여 "요약 + 분석 + 평가"가 모두 포함된 * 리뷰를 작성해 주세요.

요약 방식은 다음을 지켜 주세요:

1. 논문 개요: 논문의 주제와 핵심을 간단히 설명
2. 문제 정의: 논문이 해결하려는 문제와 기존 한계
3. 제안 방법: 논문이 제안하는 방법/모델/아이디어 설명
4. 실험 결과: 어떤 데이터로 실험했고, 성능은 어땠는지
5. 강점: 논문의 좋은 점, 혁신성, 실용성 등
6. 한계점: 아쉬운 점, 개선 가능성
7. 개인적 평가 및 의견: 내 생각과 비판적 의견 (미래 방향 제시 가능)

주의사항:
- 자연스럽고 친절하게 서술해 주세요.
- 한국어로 작성해 주세요.
- 각각의 항목(논문 개요, 문제 정의, 제안 방법, 실험 결과, 강점, 한계점, 개인적 평가)을 명확하게 소제목(예: '2. 문제 정의')을 달아 구분해 주세요.
- 소제목과 본문을 명확히 나누어 주세요.
- 자연스러운 설명 흐름을 유지하되, 항목은 반드시 구분해 주세요.
- 답변에 **굵은 글씨(Bold)** (예: **텍스트**)를 사용하지 말아주세요.
- 모든 내용은 일반 서술형 텍스트로 작성해 주세요.
- Markdown 스타일 포맷팅(굵게, 이탤릭 등)은 사용하지 말아주세요.

[논문 전문]

{content}
"""
)

refine_prompt = PromptTemplate(
    input_variables=["content"],
    template="""
다음은 논문 리뷰 초안입니다.

이 리뷰를 더 친절하고 자연스러운 튜터 스타일로 다시 작성해 주세요.  
구체적으로 다음을 지켜 주세요:

- 각 부분을 부드럽게 연결해 주세요 (단순 나열 X, 자연스러운 설명 흐름으로)
- '왜 중요한지'를 독자가 이해할 수 있도록 덧붙여 주세요.
- 핵심 개념은 짧게 정의해 주세요.
- 비판할 때는 구체적으로 (ex: "이 구조적 한계 때문에 작은 객체 탐지가 어렵다" 같은 설명)
- 마지막에는 이 논문이 가진 의의와 미래 연구 방향을 강조해 주세요.
- 너무 딱딱한 요약체가 아니라, '학생에게 설명해주는 듯한 부드러운 스타일'로 작성해 주세요.
- 반드시 인삿말(예: '안녕하세요', '오늘은 ~') 없이, 논문 개요부터 바로 시작해 주세요.
- 답변에 **굵은 글씨(Bold)** (예: **텍스트**)를 사용하지 말아주세요.
- 모든 내용은 일반 서술형 텍스트로 작성해 주세요.
- Markdown 스타일 포맷팅(굵게, 이탤릭 등)은 사용하지 말아주세요.

[리뷰 초안]

{content}
"""
)

ppt_prompt = PromptTemplate(
    input_variables=["content"],
    template="""
당신은 PPT 제작 전문가입니다.

다음은 논문 리뷰 텍스트입니다.
이 리뷰를 기반으로 논문 발표용 PPT 슬라이드를 구성해 주세요.

요구사항:
- 총 6개의 슬라이드를 추천해 주세요.
- 각 슬라이드는 "슬라이드 제목"과 "간단한 설명" 형태로 작성해 주세요.
- 각 슬라이드 주제에 어울리는 "아이콘 키워드" (예: lightbulb, flask, chart, rocket)도 '하나만' 함께 추천해 주세요.
- 포맷은 다음과 같이 해주세요:

   n번째 슬라이드 / 제목: 슬라이드 제목
   설명: 슬라이드에 들어갈 내용 요약
   아이콘 키워드: 아이콘 이름

{content}
"""
)

# LLMChains
summary_chain = LLMChain(llm=llm, prompt=summary_prompt)
review_chain = LLMChain(llm=llm, prompt=review_prompt)
refine_chain = LLMChain(llm=llm, prompt=refine_prompt)
ppt_chain = LLMChain(llm=llm, prompt=ppt_prompt)

# Icons8 매핑

def get_icon_url_from_icons8(keyword):
    icons8_library = {
        "lightbulb": "https://img.icons8.com/ios-filled/100/light-on.png",
        "flask": "https://img.icons8.com/ios-filled/100/experimental-flask-2.png",
        "chart": "https://img.icons8.com/ios-filled/100/combo-chart--v1.png",
        "rocket": "https://img.icons8.com/ios-filled/100/rocket.png",
        "idea": "https://img.icons8.com/ios-filled/100/idea.png",
        "magnifying-glass": "https://img.icons8.com/?size=100&id=7695&format=png&color=000000",
        "broken-chain": "https://img.icons8.com/?size=100&id=W7rVpJuanYI8&format=png&color=000000",
        "puzzle": "https://img.icons8.com/?size=100&id=1775&format=png&color=000000",
        "structure": "https://img.icons8.com/?size=100&id=11232&format=png&color=000000"
    }
    return icons8_library.get(keyword.lower(), icons8_library["idea"])

# Streamlit 앱

def main():
    st.title("📚 논뭉치")

    uploaded_file = st.file_uploader("논문 PDF 업로드", type=["pdf"])

    if uploaded_file:
        text = extract_text_from_pdf(uploaded_file)
        tab1, tab2, tab3, tab4 = st.tabs(["📄 논문 요약", "🔍 논문 추천", "📝 논문 리뷰", "🎨 PPT 구성 추천"])

        with tab1:
            st.subheader("논문 요약")
            if st.button("논문 요약하기"):
                with st.spinner("요약 중..."):
                    summary = summary_chain.run({"content": text})
                st.success("요약 완료!")
                st.text_area("요약 결과", summary, height=600)
                st.session_state["summary"] = summary

        with tab2:
            st.subheader("논문 추천")
            if st.button("논문 추천받기"):
                with st.spinner("추천 키워드 생성 중..."):
                    keywords = extract_keywords_tfidf(text)
                with st.spinner("논문 검색 중..."):
                    papers = search_arxiv(keywords)
                if papers:
                    for paper in papers:
                        st.markdown(f"**{paper.get('title', '제목 없음')}**")
                        st.write(f"[논문 링크]({paper.get('url', '')})")
                        with st.expander("초록 보기"):
                            st.write(paper.get("abstract", "초록 없음"))
                else:
                    st.warning("추천할 논문을 찾지 못했어요")

        with tab3:
            st.subheader("논문 리뷰 생성")
            if st.button("논문 리뷰 작성하기"):
                with st.spinner("리뷰 작성 중..."):
                    raw_review = review_chain.run({"content": text})
                    refined_review = refine_chain.run({"content": raw_review})
                st.success("리뷰 작성 완료!")
                st.text_area("최종 리뷰 결과", refined_review, height=800)
                st.session_state["refined_review"] = refined_review

        with tab4:
            st.subheader("PPT 슬라이드 구성 추천")
            if "refined_review" in st.session_state:
                if st.button("PPT 슬라이드 자동 추천받기"):
                    with st.spinner("슬라이드 추천 중..."):
                        ppt_output = ppt_chain.run({"content": st.session_state["refined_review"]})
                        slide_infos = []
                        for block in ppt_output.strip().split("\n\n"):
                            lines = block.strip().split("\n")
                            if len(lines) >= 3:
                                title = lines[0].split(":", 1)[-1].strip()
                                desc = lines[1].split(":", 1)[-1].strip()
                                icon = lines[2].split(":", 1)[-1].strip()
                                slide_infos.append({"title": title, "content": desc, "icon_keyword": icon})
                                
                    st.success("추천 완료!")
                    for slide in slide_infos:
                        st.markdown(f"### {slide['title']}")
                        st.write(slide["content"])
                        st.image(get_icon_url_from_icons8(slide["icon_keyword"]), width=80)
                        st.write(f"추천 아이콘 키워드: `{slide['icon_keyword']}`")
            else:
                st.info("논문 리뷰를 먼저 작성해 주세요.")

if __name__ == "__main__":
    main()
