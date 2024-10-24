from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain.agents.agent import AgentExecutor

import os
from dotenv import load_dotenv

import pandas as pd


class PandasAgent:

    def __init__(self, model: str):
        load_dotenv()
        self.model_name = model

        current_dir: str = os.path.dirname(__file__)
        current_dir: str = os.path.dirname(current_dir)
        current_dir: str = os.path.dirname(current_dir)
        current_dir: str = os.path.dirname(current_dir)
        # self.data_path: str = os.path.join(current_dir, "data", "data.csv")
        self.data_path: str = "data/data.csv"

        self.load_data()

        self.context: str = f"""Eres un agente encargado de responder preguntas sobre un conjunto de datos para una empresa. 
        El conjunto de datos contiene información sobre los créditos solicitados por distintas personas a la empresa, ya que es una entidad financiera. 
        Tienes acceso a un dataframe `df` de pandas, que contiene la información mencionada. Acá tienes la salida de `df.head()`:

        {self.df.head()}

        Responde en español. Solo responde con el resultado y lo que necesites para contextualizar la respuesta. No escribas código, solo el resultado."""

        models_map: dict[str, callable] = {
            "gpt-4o": lambda: self.init_openai_llm(
                "gpt-4o", os.getenv("OPENAI_API_KEY")
            ),
            "llama3-8b-8192": lambda: self.init_groq_llm(
                "llama3-8b-8192", os.getenv("GROQ_API_KEY")
            ),
        }

        models_map[self.model_name]()
        self.init_agent()

    def init_openai_llm(self, model_name: str, api_key: str):
        """
        Configure the OpenAI client using the given model name and API key.

        The temperature is set to 0 (deterministic outputs), and the maximum number of tokens
        per response is limited to 10,000.

        :param model_name: The name of the model to use.
        :type model_name: str
        :param api_key: The API key to use.
        :type api_key: str
        """

        self.llm_model: ChatOpenAI = ChatOpenAI(
            api_key=api_key,
            model=model_name,
            temperature=0,
            max_tokens=10000,
        )

    def init_groq_llm(self, model_name: str, api_key: str):
        """
        Configure the Groq client using the given model name and API key.

        The temperature is set to 0 (deterministic outputs), and the maximum number of tokens
        per response is limited to 8,192.

        :param model_name: The name of the model to use.
        :type model_name: str
        :param api_key: The API key to use.
        :type api_key: str
        """

        self.llm_model: ChatGroq = ChatGroq(
            api_key=api_key,
            model=model_name,
            temperature=0,
            max_tokens=8192,
        )

    def init_agent(self):
        """
        Initialize the agent using the model and the dataframe with the loaded data.
        """

        self.agent: AgentExecutor = create_pandas_dataframe_agent(
            llm=self.llm_model,
            df=self.df,
            agent_type="tool-calling",
            allow_dangerous_code=True,
        )

    def load_data(self):
        """
        Load the data from the CSV file into a pandas dataframe.
        """

        self.df: pd.DataFrame = pd.read_csv(self.data_path)

    def invoke(self, query: str) -> str:
        """
        Invoke the agent with the given query.

        :param query: The query to send to the agent.
        :type query: str
        :return: The response from the agent.
        :rtype: str
        """

        response: str = self.agent.invoke(
            {
                "input": f"Contexto de la información: \n{self.context.strip()}\n\n Mensaje del usuario: \n{query}"
            }
        )["output"]

        return response
