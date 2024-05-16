from setuptools import setup

setup(
    name='identity_recognition',
    version='1.0',
    author="csy",
    packages=(
        "identity_recognition",
        "identity_recognition.evaluators",
        "identity_recognition.input_pipeline",
        "identity_recognition.model",
        "identity_recognition.trainer",
        "identity_recognition.utils"),
)