import os
import json
import pandas as pd
import traceback
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.llms import HuggingFaceHub
from src.mcqgenerator.utils import read_file, get_table_data
from src.mcqgenerator.logger import logging

load_dotenv()

key = os.getenv("HUGGINGFACE_API_TOKEN")

llm = HuggingFaceHub(
    repo_id="agentica-org/DeepCoder-14B-Preview",
    huggingfacehub_api_token=key,
    task="text-generation",
    model_kwargs={
        "temperature": 0.5,
        "max_new_tokens": 512,
    }
)

# Prompt 1: Quiz Generator
TEMPLATE = """
Text: {text}
You are an expert MCQ maker. Given the above text, it is your job to \
create a quiz of {number} multiple choice questions for {subject} students in {tone} tone.
Make sure the questions are not repeated and check all the questions to be conforming the text as well.
Make sure to format your response like RESPONSE_JSON below and use it as a guide. \
Ensure to make {number} MCQs
### RESPONSE_JSON
{response_json}
"""

quiz_prompt = PromptTemplate(
    input_variables=["text", "number", "subject", "tone", "response_json"],
    template=TEMPLATE,
)

quiz_chain = LLMChain(
    llm=llm,
    prompt=quiz_prompt,
    output_key="quiz",
    verbose=True,
)

# Prompt 2: Quiz Review
TEMPLATE2 = """
You are an expert English grammarian and writer. Given a Multiple Choice Quiz for {subject} students. \
You need to evaluate the complexity of the question and give a complete analysis of the quiz. Only use at max 50 words for each analysis. \
If the quiz is not at par with the cognitive and analytical abilities of the students, \
update the quiz questions which need to be changed and change the tone such that it perfectly fits the student abilities.

MCQs_QUIZ: 
{quiz}

Check from an expert English Writer of the above quiz:
"""

review_prompt = PromptTemplate(
    input_variables=["subject", "quiz"],
    template=TEMPLATE2,
)

review_chain = LLMChain(
    llm=llm,
    prompt=review_prompt,
    output_key="review",
    verbose=True,
)

# Combine into SequentialChain
full_chain = SequentialChain(
    chains=[quiz_chain, review_chain],
    input_variables=["text", "number", "subject", "tone", "response_json"],
    output_variables=["quiz", "review"],
    verbose=True,
)
