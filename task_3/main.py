import logging
import os
import subprocess
import sys
from pathlib import Path
import click
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("main")
load_dotenv()

###############################################################################
def run_step(name: str, cmd: list[str]):
    logger.info(f"{name} started")
    res = subprocess.run([sys.executable, *cmd], check=False)
    if res.returncode != 0:
        logger.error(f"Шаг {name} упал")
        sys.exit(res.returncode)

###############################################################################
@click.command()
@click.option("--since", default=None, help="YYYY-MM-DD")
@click.option("--until", default=None, help="YYYY-MM-DD")
@click.option("--skip-parse", is_flag=True, help="Не парсить, использовать data/posts.parquet")

###############################################################################
def main(since, until, skip_parse):
    Path(os.getenv("DATA_DIR", "./data")).mkdir(parents=True, exist_ok=True)
    Path(os.getenv("OUTPUT_DIR", "./output")).mkdir(parents=True, exist_ok=True)
    if not skip_parse:
        cmd = ["parse_telegram.py"]
        if since:
            cmd += ["--since", since]
        if until:
            cmd += ["--until", until]
        run_step("Парсинг Telegram", cmd)

    run_step("Кластеризация", ["clustering.py"])
    run_step("Сентимент", ["sentiment.py"])
    run_step("Сигналы", ["signals.py"])
    run_step("Визуализация", ["visualize.py"])

###############################################################################
if __name__ == "__main__":
    main()
