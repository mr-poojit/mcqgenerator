import os
import json
import traceback
from PyPDF2 import PdfReader  # Updated import
from huggingface_hub import InferenceClient

def read_file(file):
    if file.name.endswith(".pdf"):
        try:
            text = ""
            reader = PdfReader(file)  # Updated reader
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
            return text
        except Exception as e:
            raise Exception("error reading the PDF file") from e

    elif file.name.endswith(".txt"):
        return file.read().decode("utf-8")

    else:
        raise Exception("Unsupported file format — only PDF and text files are supported.")

def get_table_data(quiz_str):
    try:
        quiz_dict = json.loads(quiz_str)  # Convert string to dict
        quiz_table_data = []

        for key, value in quiz_dict.items():
            mcq = value["mcq"]
            options = " || ".join(
                [f"{option} → {option_value}" for option, option_value in value["options"].items()]
            )
            correct = value["correct"]
            quiz_table_data.append({"MCQ": mcq, "Choices": options, "Correct": correct})

        return quiz_table_data

    except Exception as e:
        traceback.print_exception(type(e), e, e.__traceback__)
        return False
