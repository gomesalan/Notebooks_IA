# 🤖 Assistente de RH com RAG & Reranking Semântico

Uma aplicação web interativa desenvolvida com **Streamlit**, **LangChain (LCEL)** e **OpenAI**, projetada para atuar como um assistente virtual corporativo de Recursos Humanos. O agente responde a dúvidas dos colaboradores com base exclusivamente nos documentos de políticas internas da empresa, garantindo respostas precisas e alinhadas às diretrizes organizacionais.

---

## 🌟 Diferenciais do Projeto

* **Arquitetura RAG Avançada:** Combina busca vetorial por similaridade com uma etapa de **Reranking Semântico** acionada via LLM (`gpt-4o-mini`) para reordenar os trechos recuperados por relevância.
* **Organização de Dados:** Leitura automatizada de arquivos PDF armazenados na pasta `documentos/` com enriquecimento dinâmico de metadados (*férias*, *home office*, *conduta*).
* **Pipelines com LCEL:** Orquestração declarativa do fluxo de RAG utilizando a sintaxe moderna do LangChain (*LangChain Expression Language*).
* **Portabilidade Total:** Gerenciamento de caminhos dinâmicos com `os.path` para execução transparente em Windows, macOS ou Linux.

---

## 🛠️ Tecnologias Utilizadas

* **Linguagem:** Python 3.10+
* **Interface Gráfica:** Streamlit
* **Framework RAG:** LangChain (LCEL)
* **Modelos de IA:** OpenAI (`gpt-4o-mini` para geração e reranking; `text-embedding-3-small` para embeddings)
* **Vector Store:** ChromaDB
* **Processamento de PDFs:** PyMuPDF (`fitz`)

---

## ⚙️ Como Funciona o Pipeline (RAG + Reranking)

```text
[Usuário] -> Pergunta
               │
               ▼
   [1. Recuperação Vetorial] ---> Busca Top 10 Chunks no ChromaDB
               │
               ▼
   [2. Reranking Semântico]  ---> Mini-Chain LCEL pontua a relevância (Score 0-10)
               │
               ▼
   [3. Filtro de Contexto]   ---> Seleciona os Top 5 Chunks com maior pontuação
               │
               ▼
   [4. Geração da Resposta]  ---> GPT-4o-mini responde estritamente baseado no contexto

```

---

## 📂 Estrutura do Projeto

```text
.
├── documentos/              # Pasta contendo as políticas internas em PDF
│   ├── codigo_conduta.pdf
│   ├── politica_ferias.pdf
│   └── politica_home_office.pdf
├── chroma_rh/               # Diretório de persistência do ChromaDB (gerado na execução)
├── app.py                   # Script principal da aplicação
├── requirements.txt         # Dependências do projeto
├── .env                     # Variáveis de ambiente (API Keys)
└── README.md                # Documentação do projeto

```

---

## 🚀 Como Executar o Projeto Localmente

### 1. Clonar o Repositório

```bash
git clone https://github.com/seu-usuario/seu-repositorio.git
cd seu-repositorio

```

### 2. Criar e Ativar o Ambiente Virtual

```bash
# Criar ambiente virtual
python -m venv .venv

# Ativar no Windows (CMD / PowerShell)
.venv\Scripts\activate

# Ativar no Linux / macOS
source .venv/bin/activate

```

### 3. Instalar as Dependências

```bash
pip install -r requirements.txt

```

### 4. Configurar as Variáveis de Ambiente

Crie um arquivo `.env` na raiz do projeto contendo a sua chave da OpenAI:

```env
OPENAI_API_KEY=sk-proj-sua-chave-aqui

```

### 5. Executar a Aplicação

```bash
streamlit run app.py

```

Acesse o endereço exibido no terminal (geralmente `http://localhost:8501`) no seu navegador.

*Obs: texto gerado por IA (Gemini) e revisado pelo autor.