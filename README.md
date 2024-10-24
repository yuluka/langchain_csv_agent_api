# Langchain CSV Agent

## Author

Yuluka Gigante Muriel


## Overview

This repository contains the implementation of a simple Web API designed to interact with a CSV-specialized agent. The API allows users to chat with the agent, set a custom dataset, and choose different large language models (LLMs) to handle the agent's responses (gpt-4o from OpenAI, and llama3-8b-8192 from Groq).


## Endpoints

these are the available endpoints:

- `/chat/{query}`: Send a message to the agent.

    - **Param:** `query: str` the question you want to make to the agent.

    - **Response:** `response: dict[str, str]` a `dict` with the answer of the agent to the last query, with the key `bot_response`.

- `/set_llm`: Set the model the agent'll use.
    
    - **Body:** `model_name: str` the name of the model to set. You must specify one of the models in:

        ```python
        available_models: list[str] = ["gpt-4o", "llama3-8b-8192"]
        ```

        Otherwise, the agent will use the default one (`gpt-4o`).
    
    - **Response:** `response: dict[str, bool | str]` the result of the action (True if all went well, False otherwise) and the current LLM set.

- `set_data`: Set the data the agent'll be base in to answer your questions.

    - **Body:** `file: CSV File` the file with the data. It must be named as `data.csv`.

    - **Response:** If everything went well, you'll get a `dict[str, str]` with a message indicating that the action was successful. If not, you'll get an HTTP 500 error.


## How to use it

To use this project you must follow these steps:

1. Install the dependencies listed in 'requirements.txt':
    
    ```bash
    pip install -r requirements.txt
    ```

2. Get the necessary API keys:

    It is necessary to get API keys to use the LLMs. In this case, I'm using models from Groq and OpenAI, so you can get the keys on:

    - [Groq](https://console.groq.com/keys)
    - [OpenAI](https://platform.openai.com/api-keys)

3. Create a `.env` file:

    Once you have the key(s) you'll use, create a `.env` file at the root of the project, and write `OPENAI_API_KEY=«the_openai_key»` or `GROQ_API_KEY=«the_groq_key»`.

    **Note:** To use the OpenAI models, it is necessary to pay (minimum $5).

    In addition, you must create a variable called `SECURE_KEY`. This variable will contain the token that will allow you to authenticate yourself when making requests.

    Make sure to use the value of `SECURE_KEY` as the Bearer Token when making requests.

4. Collect and store the data:

    You'll need to collect the information you want your agent to use when answering the questions. It can be any information stored in a `.csv` file.

    The data must be placed in a folder called `data`, at the root of the project.

5. Run the `main.py` script:

    To start the API, open a terminal and run the following command:

    ```bash
    uvicorn src.main:app --reload
    ```

    > **Note**: The service will start, and the last message must be something like `INFO:     Application startup complete.`.

    Once the API is running, you can start making requests.
    