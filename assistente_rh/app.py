# ============================================
# PROJETO — ASSISTENTE DE RH COM RAG + RERANKING
# LangChain (LCEL) + Streamlit
# ============================================

import os
import streamlit as st
from dotenv import load_dotenv

# Carrega as variáveis de ambiente
load_dotenv()

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Configuração de diretórios usando caminhos relativos
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PERSIST_DIRECTORY = os.path.join(BASE_DIR, "chroma_rh")

# Configuração das chaves e clients 
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5, api_key=OPENAI_API_KEY)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=OPENAI_API_KEY)

# =========================
# 1. LEITURA DOS DOCUMENTOS
# =========================

# Definindo a pasta onde ficam os PDFs
DOCS_DIR = os.path.join(BASE_DIR, "documentos")

@st.cache_data
def carregar_documentos():
    caminhos_relativos = [
        "politica_ferias.pdf",
        "politica_home_office.pdf",
        "codigo_conduta.pdf"
    ]
    
    documentos = []
    for nome_arquivo in caminhos_relativos:
        # Busca o arquivo especificamente dentro da pasta 'documentos'
        caminho_completo = os.path.join(DOCS_DIR, nome_arquivo)
        
        if os.path.exists(caminho_completo):
            loader = PyMuPDFLoader(caminho_completo) 
            docs = loader.load()
            for doc in docs:
                doc.metadata["documento"] = nome_arquivo
            documentos.extend(docs)
        else:
            st.warning(f"Arquivo não encontrado em {caminho_completo}")
            
    return documentos

# =========================
# 2. CHUNKING
# =========================

def gerar_chunks(documentos):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )
    return splitter.split_documents(documentos)

# =========================
# 3. ENRIQUECIMENTO COM METADADOS
# =========================

def enriquecer_chunks(chunks):
    for chunk in chunks:
        texto = chunk.page_content.lower()
        if "férias" in texto or "ferias" in texto:
            chunk.metadata["categoria"] = "ferias"
        elif "home office" in texto or "remoto" in texto:
            chunk.metadata["categoria"] = "home_office"
        elif "conduta" in texto or "ética" in texto or "etica" in texto:
            chunk.metadata["categoria"] = "conduta"
        else:
            chunk.metadata["categoria"] = "geral"
    return chunks

# =========================
# 4. VECTOR STORE
# =========================

@st.cache_resource
def criar_vectorstore(_chunks):
    vectorstore = Chroma.from_documents(
        documents=_chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIRECTORY
    )
    return vectorstore

# =========================
# 5. RERANKING 
# =========================

def rerank_documentos(pergunta, documentos, llm):
    """
    Reordena os documentos utilizando uma mini-cadeia LCEL para scoring
    """
    prompt_rerank = ChatPromptTemplate.from_template("""
Você é um especialista em políticas internas de RH.

Pergunta do usuário:
{pergunta}

Trecho do documento:
{texto}

Avalie a relevância desse trecho para responder a pergunta.
Responda apenas com um número de 0 a 10.
""")
    
    rerank_chain = prompt_rerank | llm | StrOutputParser()
    documentos_com_score = []

    for doc in documentos:
        score_texto = rerank_chain.invoke({"pergunta": pergunta, "texto": doc.page_content})
        try:
            score = float(score_texto.strip())
        except ValueError:
            score = 0.0

        documentos_com_score.append((score, doc))

    # Ordena de forma decrescente pelo score do LLM
    documentos_ordenados = sorted(documentos_com_score, key=lambda x: x[0], reverse=True)
    
    return [doc for _, doc in documentos_ordenados]

# =========================
# 6. PIPELINE RAG COMPLETO 
# =========================

def responder_pergunta(pergunta, vectorstore):
    """
    Pipeline executado usando os conceitos de composição do LCEL
    """
    # 1. Etapa de Recuperação Inicial (Top 10)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
    documentos_recuperados = retriever.invoke(pergunta)

    # 2. Etapa de Reranking Semântico
    documentos_rerankeados = rerank_documentos(pergunta, documentos_recuperados, llm)
    
    # Seleciona os top 5 mais relevantes pós-rerank
    contexto_final = documentos_rerankeados[:5]

    # 3. Construção da cadeia de Geração final
    prompt_final = ChatPromptTemplate.from_template("""
    Você é um agente de RH corporativo.
    Responda APENAS com base nas políticas internas abaixo.

    Contexto:
    {context}

    Pergunta:
    {question}
    """)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Definição da cadeia LCEL
    rag_chain = (
        {
            "context": lambda x: format_docs(x["docs"]), 
            "question": lambda x: x["question"]
        }
        | prompt_final
        | llm
        | StrOutputParser()
    )

    # Executa a geração
    resposta_texto = rag_chain.invoke({"docs": contexto_final, "question": pergunta})

    return resposta_texto, contexto_final

# =========================
# 7. INTERFACE STREAMLIT
# =========================

st.set_page_config(page_title="Assistente de RH com RAG", layout="wide")
st.title("Assistente de RH — Políticas Internas")

pergunta = st.text_input("Digite sua pergunta sobre políticas internas de RH:")

if pergunta:
    documentos = carregar_documentos()
    
    if not documentos:
        st.error("Nenhum documento PDF foi encontrado na pasta do projeto.")
    else:
        with st.spinner("Consultando políticas internas..."):
            chunks = gerar_chunks(documentos)
            chunks = enriquecer_chunks(chunks)
            vectorstore = criar_vectorstore(chunks)

            resposta, fontes = responder_pergunta(pergunta, vectorstore)

        st.subheader("Resposta")
        st.write(resposta)

        st.subheader("Fontes utilizadas")
        for i, doc in enumerate(fontes, start=1):
            st.markdown(f"**Trecho {i}**")
            st.write(f"Documento: {doc.metadata.get('documento')}")
            st.write(f"Categoria: {doc.metadata.get('categoria')}")
            st.write(doc.page_content)
            st.divider()