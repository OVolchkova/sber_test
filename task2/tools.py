import re
import logging
import numexpr as ne
from langchain_core.tools import Tool
from rag import SiPRRetriever

logger = logging.getLogger(__name__)

###############################################################################
# Калькулятор
# Разрешаем только цифры, десятичные точки/запятые, скобки
_SAFE_RE = re.compile(r"[\d\.\,\+\-\*\/\(\)\s\^\%]+")

###############################################################################
# заменяет десятичные запятые на точки
# удаляет пробелы для тысяч
# заменяет ^ на степень **
def _normalise_expression(expr):
    s = expr.strip()
    s = re.sub(r"(?<=\d) (?=\d)", "", s)
    s = s.replace(",", ".")
    s = s.replace("^", "**")
    return s

###############################################################################
# вычисляет арифметическое выражение
def calculator(expression):
    raw = expression
    expr = _normalise_expression(expression)

    # Проверяем допустимые символы
    if not _SAFE_RE.fullmatch(expr.replace("**", "")):
        return f'Ошибка: выражение "{raw}" содержит недопустимые символы'

    try:
        result = ne.evaluate(expr)
        if hasattr(result, "item"):
            result = result.item()

    except Exception as exc:
        return f"Ошибка вычисления {raw}: {exc}"

    if isinstance(result, float):
        if result.is_integer():
            return str(int(result))
        return f"{result:.6g}"
    return str(result)

###############################################################################
# инструмент калькулятора
def make_calculator_tool():
    return Tool(
        name="calculator",
        func=calculator,
        description=(
            "Безопасный калькулятор для арифметики. "
            "Принимает строку с одним выражением: цифры, +, -, *, /, **, %, ()"
            "Используй его ВСЕГДА, когда в ответе пользователю нужно получить число "
            "из нескольких чисел: сложение прогнозов, рост в процентах, среднее"))

###############################################################################
# инструмент поиска по документу
def make_retriever_tool(retriever):
    def _run(query):
        logger.info(f"knowledge_base query: {query}")
        return retriever.retrieve(query)

    return Tool(
        name="knowledge_base",
        func=_run,
        description=(
            "Поиск по официальному документу «Схема и программа развития электроэнергетических систем России на 2025-2030 годы» (СО ЕЭС)"
            "Используй его, чтобы получить факты, цифры и цитаты по: прогнозам потребления и производства электроэнергии, балансу мощности, вводам и выводам генерации, сетевым проектам, ОЭС"
            "Запрос - по-русски, осмысленной фразой"
            "(например: 'прогноз электропотребления ОЭС Юга 2027 2028')"))
