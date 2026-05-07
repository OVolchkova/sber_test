import asyncio
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List
from aiogram import Bot, Dispatcher, F
from aiogram.filters import Command, CommandStart
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import Message
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from agent import build_agent_executor, validate_api_key
from prompts import GREETING_AFTER_KEY, WELCOME_PROMPT
from rag import SIPR_PDF_URL, build_or_load_index, download_pdf
from aiogram.client.session.aiohttp import AiohttpSession

###############################################################################
# Конфигурация
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("bot")

load_dotenv()

SESSIONS = {}
RETRIEVER = None

###############################################################################
# состояния бота
# waiting_key - ждём, пока пользователь пришлёт API-ключ
# in_dialog - пользователь может задавать вопросы
class DialogStates(StatesGroup):
    waiting_key = State()
    in_dialog = State()

###############################################################################
# Класс для хранения данных одного пользователя
@dataclass
class UserSession:
    history: List[Dict[str, str]] = field(default_factory=list)
    api_key: str = None

###############################################################################
# получает сессию пользователя. если пользователя нет - создаёт новую
def get_session(user_id):
    if user_id not in SESSIONS:
        SESSIONS[user_id] = UserSession()
    return SESSIONS[user_id]

###############################################################################
# Инициализируем диспетчера с хранилищем состояний в памяти
dp = Dispatcher(storage=MemoryStorage())

###############################################################################
# то как бот отвечает на команду старт (декоратор - CommandStart())
# если он помнит сессию - он предложит продолжить, если нет - то начнет заново с запроса API
@dp.message(CommandStart())
async def on_start(msg, state):
    sess = get_session(msg.from_user.id)
    if sess.api_key:
        await msg.answer(
            "С возвращением! Ваш API-ключ уже в памяти."
            "Можете задать вопрос, или /reset для сброса истории, "
            "/key - чтобы сменить ключ."
        )
        await state.set_state(DialogStates.in_dialog)
    else:
        await msg.answer(WELCOME_PROMPT)
        await state.set_state(DialogStates.waiting_key)

###############################################################################
# команда reset - очистить историю
@dp.message(Command("reset"))
async def on_reset(msg, state):
    sess = get_session(msg.from_user.id)
    sess.history.clear()
    await msg.answer("История диалога очищена.")

###############################################################################
# команда key - когда надо сменить ключ
@dp.message(Command("key"))
async def on_change_key(msg, state):
    sess = get_session(msg.from_user.id)
    sess.api_key = None
    sess.executor = None
    sess.history.clear()
    await msg.answer("Хорошо, пришлите новый API-ключ GigaChat.")
    await state.set_state(DialogStates.waiting_key)

###############################################################################
# cрабатывает ТОЛЬКО когда пользователь в состоянии waiting_key
# проверяем подходит ли ключ
# если подходит создаем агента
@dp.message(DialogStates.waiting_key, F.text)
async def on_api_key(msg, state):
    api_key = msg.text.strip()
    sess = get_session(msg.from_user.id)

    if len(api_key) < 20:
        await msg.answer("Это не похоже на ключ. Ключ GigaChat обычно длиннее. Попробуйте ещё раз.")
        return

    await msg.answer("Проверяю ключ, секунду…")
    scope = os.getenv("GIGACHAT_SCOPE", "GIGACHAT_API_PERS")
    verify_ssl = os.getenv("GIGACHAT_VERIFY_SSL_CERTS", "false").lower() == "true"

    ok = await validate_api_key(api_key, scope, verify_ssl)
    if not ok:
        await msg.answer(
            "Не удалось получить токен по этому ключу. "
            "Проверьте, что вы скопировали Authorization Key из личного "
            "кабинета (а не client_id), и пришлите ещё раз."
        )
        return

    sess.api_key = api_key
    sess.executor = build_agent_executor(api_key, RETRIEVER)
    await msg.answer(GREETING_AFTER_KEY)
    await state.set_state(DialogStates.in_dialog)

###############################################################################
# когда пользователь в состоянии dialog 
@dp.message(DialogStates.in_dialog, F.text)
async def on_question(msg, state):
    sess = get_session(msg.from_user.id)
    if sess.executor is None:
        await msg.answer(
            "Похоже, ваш сеанс сбросился. Отправьте /key, чтобы заново"
            "ввести API-ключ."
        )
        await state.set_state(DialogStates.waiting_key)
        return

    await msg.bot.send_chat_action(msg.chat.id, "typing")
    try:
        # AgentExecutor поддерживает invoke / ainvoke
        result = await asyncio.to_thread(sess.executor.invoke, {"input": msg.text, "chat_history": sess.history})
        answer = result.get("output") or "Не удалось сформировать ответ."
    except Exception as exc:
        logger.exception("agent failed")
        answer = (
            "Произошла ошибка при обращении к GigaChat: "
            f"`{type(exc).__name__}: {exc}`. Попробуйте ещё раз или /reset.")
        await msg.answer(answer)
        return

    # Дописываем в историю - для сохранения контекста между ходами
    sess.history.append(HumanMessage(content=msg.text))
    sess.history.append(AIMessage(content=answer))
    # Не даём истории расти бесконечно
    if len(sess.history) > 20:
        sess.history = sess.history[-20:]

    # Telegram режет сообщения > 4096 символов
    for part in chunked(answer, 3500):
        await msg.answer(part, parse_mode=None)

###############################################################################
# Если состояние не установлено он предлагает отправить /start
@dp.message(F.text)
async def fallback(msg, state):
    await msg.answer("Чтобы начать, отправьте /start.")

###############################################################################
# Вспомогательная функция - разбивает строку на чанки по n символов 
def chunked(text, n):
    for i in range(0, len(text), n):
        yield text[i:i + n]

###############################################################################
# Main
async def main():
    global RETRIEVER

    tg_token = os.getenv("TG_TOKEN")
    if not tg_token:
        raise RuntimeError("TG_TOKEN is not set in env")

    pdf_path = Path(os.getenv("SIPR_PDF_PATH", "./data/sipr_ups_2025-30_fin.pdf"))
    index_path = Path(os.getenv("VECTOR_STORE_PATH", "./vector_store"))
    embed_model = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-base")
    download_pdf(SIPR_PDF_URL, pdf_path)
    RETRIEVER = build_or_load_index(pdf_path, index_path, embed_model_name=embed_model)

    # Запуск ботаtype .env
    session = AiohttpSession(proxy="http://proxy_user:fri33ghte@144.31.233.89:31228")
    bot = Bot(token=tg_token, session=session)
    logger.info("Бот запущен")
    await dp.start_polling(bot)

###############################################################################
if __name__ == "__main__":
    asyncio.run(main())
