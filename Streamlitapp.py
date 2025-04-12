import os
import json
import pandas as pd
import traceback
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from src.mcqgenerator.utils import read_file, get_table_data
from src.mcqgenerator.logger import logging
import streamlit as st

client = InferenceClient(model="mistralai/Mixtral-8x7B-Instruct-v0.1")

with open(r'C:\Users\pooji\mcqgenerator\Response.json', 'r') as file:
    RESPONSE_JSON = json.load(file)

st.title("MCQ Creator Application with Langchain")

with st.form("user_input"):
    uploaded_file = st.file_uploader("Upload a PDF or Text file")
    mcq_count = st.number_input("No. of MCQ's", min_value=3, max_value=50)
    subject = st.text_input("Insert Subject", max_chars=20)
    tone = st.text_input("Complexity Level of Questions", max_chars=20, placeholder="Simple")
    button = st.form_submit_button("Create MCQ's")

    if button and uploaded_file is not None and mcq_count and subject and tone:
        with st.spinner("Loading..."):
            try:
                text = read_file(uploaded_file)
                response = client.text_generation({
                    "text": text,
                    "parmeters": {
                    "number": mcq_count,
                    "subject": subject,
                    "tone": tone,
                    "response_json": json.dumps(RESPONSE_JSON)
                    }
                })

                if isinstance(response, dict):
                    quiz = response.get("quiz", None)
                    if quiz is not None:
                        table_data = get_table_data(quiz)
                        if table_data is not None:
                            df = pd.DataFrame(table_data)
                            df.index = df.index + 1
                            st.table(df)
                            st.text_area(label="Review", value=response["review"])
                    else:
                        st.error("Error in the table data")
                else:
                    st.write(response)

            except Exception as e:
                traceback.print_exception(type(e), e, e.__traceback__)
                st.error("An error occurred while generating MCQs.")
