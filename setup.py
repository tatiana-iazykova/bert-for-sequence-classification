from setuptools import setup, find_packages

setup(
    name="bert-for-sequence-classification",
    version='0.0.4',
    author="Tatiana Iazykova",
    author_email="tania_yazykova@bk.ru",
    description='Easy fine-tuning for BERT models',
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'bert-clf-train = bert_clf.pipeline:main',
        ],
    },
    install_requires=[
        'transformers>=4.2.0',
        'torch>=1.7.1, != 1.10',
        'numpy>=1.19.5',
        'pandas>=1.1.5',
        'scikit-learn>=1.0',
        'pyyaml>=6.0',
        'openpyxl>=3.0.9',
    ],

    keywords=['python', 'bert', 'deep learning', 'nlp'],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ]
)

