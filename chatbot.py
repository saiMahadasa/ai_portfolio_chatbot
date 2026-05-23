import logging
from pathlib import Path

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq

from config import settings

logger = logging.getLogger(__name__)

_PROMPT_TEMPLATE = """\
You are an AI assistant for Sai Subrahmanyam Mahadasa's professional portfolio.

Use the resume below to answer the visitor's question. Respond the way \
a confident, senior engineer would naturally talk about their own work — direct, \
clear, and specific. No filler, no fluff.

Resume:
{context}

Rules:
- Only use information present in the resume above. If something isn't there, say so honestly.
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


def _build_chain(resume_text: str):
    prompt = ChatPromptTemplate.from_template(_PROMPT_TEMPLATE)
    llm = ChatGroq(
        model=settings.model_name,
        api_key=settings.groq_api_key,
        temperature=settings.model_temperature,
    )
    # Resume is injected as static context on every call — it's small enough to fit easily
    return (
        {"context": lambda _: resume_text, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )


logger.info("Loading resume from '%s'.", settings.resume_path)
_resume_text = _load_resume(settings.resume_path)

_chain = _build_chain(_resume_text)
logger.info("Chatbot ready — model: %s", settings.model_name)


def answer(question: str) -> str:
    return _chain.invoke(question)
