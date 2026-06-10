# Information Retrieval Project

## Overview

This project implements core Information Retrieval (IR) techniques, including Boolean search and text preprocessing. It covers document collection from web sources, stopword removal using both predefined lists and frequency-based methods, and Boolean query processing over a document corpus.

## Requirements

* Python 3.12
* Virtual environment support (`.venv`)

## Installation

### 1. Create and Activate a Virtual Environment

Create the virtual environment:

```bash
python3 -m venv .venv
```

or

```bash
python -m venv .venv
```

Activate the environment:

```bash
source .venv/bin/activate
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Running the Project

### Run the Test Suite

```bash
pytest public_tests/
```

### Launch the User Interface

```bash
python main.py
```

## Project Structure

```text
ir_pr02_template_v1.3/
├── main.py
├── document.py
├── my_module.py
├── README.md
├── test_wrapper.txt
└── CHANGELOG.txt

public_tests/
├── test_pr02_t2.py
├── test_pr02_t3.py
├── test_pr02_t4.py
└── englishST.txt
```

