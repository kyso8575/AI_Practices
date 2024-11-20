# 환경 변수
from dotenv import load_dotenv
import os
import datetime

# AI 모델
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter

from langchain_openai import OpenAIEmbeddings
import faiss
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

load_dotenv()  # .env 파일에서 환경 변수 로드

api_key = os.getenv("OPENAI_API_KEY")

if api_key is None:
    raise ValueError("API key not found. Please set the OPENAI_API_KEY environment variable.")


model = ChatOpenAI(model="gpt-4o-mini")
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")


prompt = 'prompt2'
prompt_file_path = f'./prompts/{prompt}.txt'

# 텍스트 파일 읽기
with open(prompt_file_path, 'r', encoding='utf-8') as file:
    prompt_content = file.read()

pdf_file_path ='./docs/인공지능산업최신동향 2024년 11월.pdf'

loader = PyPDFLoader(pdf_file_path)

docs = loader.load()


# 고정된 크기로 텍스트를 분할하는 클래스
# text_splitter = CharacterTextSplitter(
#     separator="\n\n",
#     chunk_size=100,
#     chunk_overlap=10,
#     length_function=len,
#     is_separator_regex=False,
# )

# splits = text_splitter.split_documents(docs)


# 문맥을 고려하여 분할하는 클래스
recursive_text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=10,
    length_function=len,
    is_separator_regex=False,
)

splits = recursive_text_splitter.split_documents(docs)

# print(splits)

vector_store = FAISS.from_documents(documents=splits, embedding=embeddings)

retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 1})


messages = [
    ("system", line.split(": ", 1)[1])
    for line in prompt_content.splitlines()
    if line.startswith("system:") or line.startswith("user:")
]

print(messages)

# ChatPromptTemplate 생성
contextual_prompt = ChatPromptTemplate.from_messages(messages)

class DebugPassThrough(RunnablePassthrough):
    def invoke(self, *args, **kwargs):
        output = super().invoke(*args, **kwargs)
        print("Debug Output:", output)
        return output
        
# 문서 리스트를 텍스트로 변환하는 단계 추가
class ContextToText(RunnablePassthrough):
    def invoke(self, inputs, config=None, **kwargs):  # config 인수 추가
        # context의 각 문서를 문자열로 결합
        context_text = "\n".join([doc.page_content for doc in inputs["context"]])
        return {"context": context_text, "question": inputs["question"]}

# RAG 체인에서 각 단계마다 DebugPassThrough 추가
rag_chain_debug = {
    "context": retriever,                    # 컨텍스트를 가져오는 retriever
    "question": DebugPassThrough()        # 사용자 질문이 그대로 전달되는지 확인하는 passthrough
}  | DebugPassThrough() | ContextToText()|   contextual_prompt | model


result_dir = './Results'
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")  # 타임스탬프 형식: 20231118_152345
file_name = f"{prompt}_{timestamp}.txt"  # 예시 파일 이름: AI_Trend_Analysis_20231118_152345.txt

# 파일 경로 설정
file_path = os.path.join(result_dir, file_name)

# 질문과 응답을 저장할 파일 열기
with open(file_path, 'w') as file:
    while True:
        print("========================")
        query = input("질문을 입력하세요    : ")
        
        # RAG 체인에서 응답을 받음
        response = rag_chain_debug.invoke(query)
        
        # 질문과 응답을 출력
        print("Final Response:")
        print(response.content)
        
        # 질문과 응답을 파일에 기록
        file.write(f"Question: {query}\n")
        file.write(f"Response: {response.content}\n")
        file.write("========================\n")

# 챗봇이 보통 최신 정보는 학습이 되어있지 않기 때문에 RAG를 통해서, 정보를 제공해주고, 최신 정보에 대한 답을 얻을 수 있기 때문에 RAG가 중요하다.
