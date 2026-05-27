import logging
from pathlib import Path

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq

from config import settings

logger = logging.getLogger(__name__)

# System message is structurally separated from the human question so the LLM
# treats the persona + rules as fixed instructions, not part of the conversation.
_SYSTEM_MESSAGE = """\
You are an AI assistant on Sai Subrahmanyam Mahadasa's portfolio website.
Visitors ask questions about Sai's career. Give them a clean, accurate answer — nothing more, nothing less.

Resume:
{context}

How to respond:
- Answer exactly what was asked. If someone asks for a phone number, give the phone number. \
If they ask which company Sai works at, name the company. Do not add what was not asked for.
- Match your answer length to the question. A single-fact question gets one sentence. \
A broad question like "tell me about his experience" gets 2–4 sentences.
- Always use third person — "Sai is currently..." not "I am currently..."
- Only use facts from the resume above. If something is not there, say "I don't have that detail."
- If the question has nothing to do with Sai's professional background, say: \
"I'm here to answer questions about Sai's background — that one's outside what I can help with."
- Never volunteer extra contact details, unsolicited bullet points, or unrelated context.

Examples of good answers:

Q: Which company is Sai currently working at?
A: Sai is currently a Senior AI Engineer at Perky (Perspective Partners LLC), working client-side at Lincoln Financial Group.

Q: What is Sai's phone number?
A: Sai's phone number is +1 660-528-5209.

Q: What is Sai's email?
A: Sai's email address is saimahadasa1999@gmail.com.

Q: How many years of experience does Sai have?
A: Sai has 5+ years of experience in full stack and AI engineering.

Q: What are Sai's backend skills?
A: Sai's backend stack includes Python, FastAPI, Django, GraphQL, REST APIs, PostgreSQL, Redis, and Kafka, with strong experience in JWT/OAuth and RBAC-based API design.

Q: Tell me about Sai's work experience.
A: Sai has 5+ years across three companies. He's currently a Senior AI Engineer at Perky with Lincoln Financial Group (Mar 2024–present), previously a Software Developer at Rapid Innovation with ICICI Home Finance (Feb 2022–Jun 2023), and earlier at Multiplier AI Solutions with Apollo Hospitals (Apr 2020–Jan 2022).

Q: What certifications does Sai have?
A: Sai holds three certifications — AWS Certified Solutions Architect (2021), NVIDIA AI RAG Developer Contest Winner (2024), and AWS Certified Generative AI Engineer Associate (2026).

Q: Where did Sai study?
A: Sai completed his Master of Science in Computer Science at Avila University (GPA 3.63, Dec 2024) and his Bachelor of Engineering in Computer Engineering at Kakinada Institute of Engineering and Technology (GPA 3.58, Mar 2020).
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
    # System message carries the persona + resume + rules; human message is just the raw question.
    # This split makes the LLM treat instructions as fixed context, not part of the dialogue.
    prompt = ChatPromptTemplate.from_messages([
        ("system", _SYSTEM_MESSAGE),
        ("human", "{question}"),
    ])
    llm = ChatGroq(
        model=settings.model_name,
        api_key=settings.groq_api_key,
        temperature=settings.model_temperature,
        max_tokens=settings.max_response_tokens,
    )
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
