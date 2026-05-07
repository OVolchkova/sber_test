import logging
import os
from pathlib import Path
from dotenv import load_dotenv
from rag import SIPR_PDF_URL, build_or_load_index, download_pdf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("build_index")

###############################################################################
def main():
    load_dotenv()
    pdf_path = Path(os.getenv("SIPR_PDF_PATH", "./data/sipr_ups_2025-30_fin.pdf"))
    index_path = Path(os.getenv("VECTOR_STORE_PATH", "./vector_store"))
    embed_model = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-base")
    download_pdf(SIPR_PDF_URL, pdf_path)
    build_or_load_index(pdf_path, index_path, embed_model_name=embed_model, rebuild=True)
    logger.info(f"Индекс готов")

###############################################################################
if __name__ == "__main__":
    main()
