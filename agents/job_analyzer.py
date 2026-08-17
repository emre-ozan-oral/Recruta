"""Job Analyzer agent: parses the job posting into structured requirements."""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate

from llm import get_structured_llm
from schemas import JobRequirements
from state import AgentState

SYSTEM_PROMPT = """You are a meticulous technical recruiter.
Read the job posting below and extract a structured summary of what it
actually requires. Keep every field grounded in the text; don't invent
skills that aren't implied by the posting.

Classify carefully into three separate buckets — this distinction matters
a lot downstream, so don't blur it:

1. required_skills — HARD, technical, verifiable requirements only:
   specific languages/tools/frameworks, years of experience, certifications,
   degrees. Something you could check a CV against and get an objective
   yes/no.
2. soft_skills — interpersonal traits, culture-fit language, and
   passion/motivation statements (e.g. "team player", "passion for mobile
   puzzle games", "self-starter", "thrives in ambiguity"). These are NOT
   verifiable from a CV — never put them in required_skills, and never
   phrase them in a way that implies a candidate can be checked against
   them.
3. nice_to_have_skills — explicitly optional/bonus technical skills.

Don't inflate nice-to-haves or soft skills into hard requirements.

Also separately extract years_of_experience_required: the years of
professional experience the posting asks for, written exactly as stated
(e.g. "3-5 years", "5+ years"). If the posting never states a number of
years anywhere, set this to exactly "Not specified in the posting" — never
guess or infer a number, and never leave it blank."""

USER_PROMPT = "Job posting:\n\n{job_posting_text}"


def run(state: AgentState) -> dict:
    llm = get_structured_llm(JobRequirements, tier="content")
    prompt = ChatPromptTemplate.from_messages(
        [("system", SYSTEM_PROMPT), ("human", USER_PROMPT)]
    )
    chain = prompt | llm
    result: JobRequirements = chain.invoke({"job_posting_text": state["job_posting_text"]})

    return {
        "job_requirements": result.model_dump(),
        "messages": [*state.get("messages", []), "job_analyzer: extracted job requirements."],
    }
