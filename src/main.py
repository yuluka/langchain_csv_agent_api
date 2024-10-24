from fastapi import FastAPI, File, UploadFile, Depends, HTTPException, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from dotenv import load_dotenv
import os
from pydantic import BaseModel

from .model.chain.pandas_agent import PandasAgent


class ChatRequest(BaseModel):
    query: str


load_dotenv()
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

available_models: list[str] = ["gpt-4o", "llama3-8b-8192"]

selected_model: str = available_models[0]

pandas_agent: PandasAgent = PandasAgent(selected_model)

SECURE_KEY: str = os.getenv("SECURE_KEY")
security: HTTPBearer = HTTPBearer()


def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)):
    """
    Verify the token sent in the request.

    :param credentials: The credentials sent in the request.
    :type credentials: HTTPAuthorizationCredentials
    :return: Whether the token is valid.
    :rtype: bool
    """

    token = credentials.credentials

    if token != SECURE_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

    return True


@app.post("/chat")
def chat(chat_request: ChatRequest, isAuthorized: bool = Depends(verify_token)) -> dict[str, str]:
    """
    Endpoint to chat with the selected model.

    :param query: The query to send to the model.
    :type query: str
    :param isAuthorized: Whether the request is authorized.
    :type isAuthorized: bool
    :return: The response from the model.
    :rtype: dict[str, str]
    """

    return {"bot_response": pandas_agent.invoke(chat_request.query)}


@app.post("/set_llm")
def set_language_model(
    model_name: str, isAuthorized: bool = Depends(verify_token)
) -> dict[str, bool | str]:
    """
    Endpoint to set the language model to use.

    :param model_name: The name of the model to use.
    :type model_name: str
    :param isAuthorized: Whether the request is authorized.
    :type isAuthorized: bool
    :return: Whether the model was set successfully and the current selected model.
    :rtype: dict[str, bool | str]
    """

    global selected_model, pandas_agent

    selected_model = (
        model_name if model_name in available_models else available_models[0]
    )
    pandas_agent = PandasAgent(selected_model)

    return {
        "result": model_name in available_models,
        "current_selected_model": selected_model,
    }


@app.post("/set_data")
def set_data(
    file: UploadFile = File(...), isAuthorized: bool = Depends(verify_token)
) -> dict[str, str]:
    """
    Endpoint to set the data to use for the agent.

    :param file: The CSV file containing the data.
    :type file: UploadFile
    :param isAuthorized: Whether the request is authorized.
    :type isAuthorized: bool
    :return: The result of loading the data.
    :rtype: dict[str, str]
    """

    try:
        with open("data/data.csv", "wb") as data_file:
            data_file.write(file.file.read())

        pandas_agent.load_data()
        pandas_agent.init_agent()

        return {"message": "Data loaded successfully!"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
