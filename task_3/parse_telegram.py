import asyncio
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlparse
import click
import pandas as pd
from dotenv import load_dotenv
from telethon import TelegramClient
from tqdm import tqdm

###############################################################################
# Конфигурация
###############################################################################

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("parser")
load_dotenv()

###############################################################################
# Прокси
# Я использую личный прокси, пожалуйста, не используйте его для других целей

PROXY_CONFIG = {
    "url": os.getenv("PROXY_URL"),
    "username": os.getenv("PROXY_USERNAME"),
    "password": os.getenv("PROXY_PASSWORD"),
}

###############################################################################
# функция для конвевртации proxy в формат словаря для использования в библиотеке telephone
def get_proxy():
    parsed = urlparse(PROXY_CONFIG["url"])

    return {
        "proxy_type": parsed.scheme,
        "addr": parsed.hostname,
        "port": parsed.port,
        "username": PROXY_CONFIG.get("username"),
        "password": PROXY_CONFIG.get("password"),
        "rdns": True,
    }

###############################################################################
# Окно дат
def default_window():
    today = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    return today - timedelta(days=30), today

###############################################################################
def parse_date(s):
    return datetime.fromisoformat(s).replace(tzinfo=timezone.utc)

###############################################################################
# Парсинг сообщений
async def fetch_messages(channel, since, until, api_id, api_hash, phone, session_name):
    rows = []
    client = TelegramClient(session_name, api_id, api_hash, proxy=get_proxy(), timeout=20, connection_retries=2)
    await client.start(phone=phone)
    me = await client.get_me()
    logger.info(f"Подключились как @{me.username}")
    entity = await client.get_entity(channel)
    pbar = tqdm(desc="messages", unit="msg")
    async for msg in client.iter_messages(entity, offset_date=until, reverse=False):
        if msg.date < since:
            break

        text = (msg.message or "").strip()
        rows.append({"id": msg.id, "date": msg.date.astimezone(timezone.utc), "text": text})
        pbar.update(1)

    pbar.close()
    await client.disconnect()
    df = pd.DataFrame(rows)
    if df.empty:
        logger.warning("Нет сообщений за период")
        return df
    return df.sort_values("date").reset_index(drop=True)


###############################################################################
@click.command()
@click.option("--since", default=None, help="Дата начала (YYYY-MM-DD)")
@click.option("--until", default=None, help="Дата конца (YYYY-MM-DD)")
@click.option("--channel", default=None, help="Имя канала без @")
@click.option("--out", default=None, help="Куда сохранить parquet")

def main(since, until, channel, out):
    api_id = int(os.environ["TG_API_ID"])
    api_hash = os.environ["TG_API_HASH"]
    phone = os.getenv("TG_PHONE")
    session = os.getenv("TG_SESSION_NAME", "markettwits")
    channel = channel or os.getenv("TG_CHANNEL", "markettwits")
    data_dir = Path(os.getenv("DATA_DIR", "./data"))
    data_dir.mkdir(parents=True, exist_ok=True)
    out = Path(out) if out else data_dir / "posts.parquet"
    d_since, d_until = default_window()
    since_dt = parse_date(since) if since else d_since
    until_dt = parse_date(until) if until else d_until

    df = asyncio.run(fetch_messages(channel, since_dt, until_dt, api_id, api_hash, phone, session))
    if df.empty:
        return
    df.to_parquet(out, index=False)

###############################################################################
if __name__ == "__main__":
    main()