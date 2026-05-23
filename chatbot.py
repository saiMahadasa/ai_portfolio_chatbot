import logging
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import settings

logger = logging.getLogger(__name__)

_PROMPT_TEMPLATE = """\
You are an AI assistant for Sai Subrahmanyam Mahadasa's professional portfolio.

Use the resume excerpts below to answer the visitor's question. Respond the way \
a confident, senior engineer would naturally talk about their own work — direct, \
clear, and specific. No filler, no fluff.

Resume context:
{context}

Rules:
- Only use information present in the context above. If something isn't there, say so honestly.
- Keep it conversational: 2–4 sentences for simple questions, more only when genuine detail is needed.
- Refer to Sai in the third person ("Sai currently works at...", "His background spans...").
- If the question has nothing to do with Sai's professional background, reply with: \
"I'm here to answer questions about Sai's professional background — that one's a bit outside my scope!"
- Never fabricate job titles, company names, dates, or metrics.

Question: {question}
"""


def _load_resume(path: str) -> str:
    resume_file = Path(path)
    if not resume_file.exists():
        raise FileNotFoundError(
            f"Resume file not found at '{path}'. Check RESUME_PATH in your .env."
        )
    text = resume_file.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Resume file at '{path}' is empty.")
    return text


def _build_vectorstore(text: str) -> FAISS:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " "],
    )
    chunks = splitter.split_text(text)
    logger.info("Resume split into %d chunks.", len(chunks))

    # Embeddings run locally — no API call, no cost
    embeddings = HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    store = FAISS.from_texts(chunks, embeddings)
    logger.info("FAISS index ready.")
    return store


def _format_docs(docs) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


def _build_chain(vectorstore: FAISS):
    retriever = vectorstore.as_retriever(search_kwargs={"k": settings.retriever_k})
    prompt = ChatPromptTemplate.from_template(_PROMPT_TEMPLATE)
    llm = ChatGroq(
        model=settings.model_name,
        api_key=settings.groq_api_key,
        temperature=settings.model_temperature,
    )
    return (
        {"context": retriever | _format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )


# Build the index and chain once at startup, reuse across all requests
logger.info("Loading resume from '%s'.", settings.resume_path)
_resume_text = _load_resume(settings.resume_path)

logger.info("Building vector store…")
_vectorstore = _build_vectorstore(_resume_text)

_chain = _build_chain(_vectorstore)
logger.info("Chatbot ready — %s / %s", settings.model_name, settings.embedding_model)


def answer(question: str) -> str:
    return _chain.invoke(question)
