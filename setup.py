from setuptools import find_packages,setup

setup(
    name='mcqgenerator',
    version='0.0.1',
    author='mr-poojit',
    author_email='princepoojit123@gmail.com',
    install_requires=["openai","huggingface","langchain","streamlit","python-dotenv","PyPDF2"],
    packages=find_packages()
)