import logging
import numexpr
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools import StructuredTool
from langchain_core.prompts import ChatPromptTemplate
from langchain_gigachat import GigaChat
from pydantic import BaseModel, Field
from prompts import SYSTEM_PROMPT
from rag import retrieve_sipr

logger = logging.getLogger(__name__)

###############################################################################
# инструмент для поиска по базе данных
class KnowledgeBaseInput(BaseModel):
    query: str = Field(description="Поисковый запрос для документа СиПР ЭЭС России 2025-2030")

###############################################################################
# ищет 5 наиболее релевантных кусков документа по запросу
def make_kb_tool(vectorstore):
    def search_kb(query):
        return retrieve_sipr(vectorstore, query, k=5)

    return StructuredTool.from_function(
        func=search_kb,
        name="knowledge_base",
        description=(
            "Поиск информации в документе СиПР ЭЭС России 2025-2030. "
            "Используй для вопросов о планах развития энергетики, "
            "мощностях электростанций, объектах генерации, "
            "балансе мощности и электроэнергии в России."
        ),
        args_schema=KnowledgeBaseInput,
    )

###############################################################################
# инструмент калькулятор
class CalculatorInput(BaseModel):
    expression: str = Field(description="Математическое выражение, например '6400 * 8760 * 0.8'")

###############################################################################
def calculate(expression):
    try:
        result = numexpr.evaluate(expression).item()
        return str(result)
    except Exception as e:
        return f"Ошибка вычисления: {e}"


calculator_tool = StructuredTool.from_function(
    func=calculate,
    name="calculator",
    description=(
        "Калькулятор для математических выражений."
        "Используй для расчётов мощности, годовой выработки энергии, КИУМ."
        "Принимает строку с выражением, возвращает число."
        "Пример: '6400 * 8760 * 0.8' для расчёта годовой выработки в МВт/ч."
    ),
    args_schema=CalculatorInput,
)

###############################################################################
# Основная функция по созданию агента
def build_agent_executor(api_key, vectorstore):
    llm = GigaChat(
        credentials=api_key,
        scope="GIGACHAT_API_PERS",
        verify_ssl_certs=False,
        model="GigaChat-Max",
    )

    tools = [make_kb_tool(vectorstore), calculator_tool]

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    agent = create_tool_calling_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools, verbose=True, max_iterations=5, return_intermediate_steps=False)

    return executor

###############################################################################
# Функция валидации API ключа
async def validate_api_key(api_key, scope, verify_ssl):
    llm = GigaChat(credentials=api_key, scope=scope, model="GigaChat", verify_ssl_certs=verify_ssl)
    try:
        await llm.ainvoke("ok")
        return True
    except Exception as exc:
        logger.warning("validate_api_key failed: %s", exc)
        return False