from setuptools import setup, find_packages


setup(
    name="sciriff",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas==2.2.2",
        "pyyaml==6.0.1",
        "jinja2==3.1.4",
        "black==24.4.2",
        "nltk==3.8.1",
        "rouge_score==0.1.2",
        "bioc==2.1",
        "huggingface-hub==0.23.3",
        "jsonschema==4.22.0",
        "beaker-py",
        "spacy==3.7.4",
        "openai==1.47.0",
        "thefuzz==0.20.0",
        "pytest",
        "tqdm",
        "lm-eval==0.4.4"
    ],
    include_package_data=True,
)
