import os
import logging
from pathlib import Path
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

###############################################################################
logger = logging.getLogger(__name__)

SIPR_PDF_URL = (
    "https://www.so-ups.ru/fileadmin/files/company/future_plan/"
    "public_discussion/2025-30_final/sipr_ups_2025-30_fin.pdf"
)

###############################################################################
# скачивает pdf если его еще нет на диске
def download_pdf(url, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        logger.info(f"PDF уже скачан")
        return dest

    import urllib.request
    logger.info(f"Скачиваю {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (compatible; SiPR-RAG/1.0)"})
    with urllib.request.urlopen(req, timeout=120) as resp, open(dest, "wb") as f:
        f.write(resp.read())
    logger.info(f"Готово")
    return dest

###############################################################################
def load_and_split(pdf_path, chunk_size = 800, chunk_overlap = 150):
    pages = PyPDFLoader(str(pdf_path)).load()
    logger.info(f"Страниц в документе: {len(pages)}")
    for p in pages:
        p.metadata["source"] = "СиПР ЭЭС России 2025-2030"
        p.metadata["page"] = p.metadata.get("page", 0) + 1

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", "; ", ", ", " ", ""],
    )
    docs = splitter.split_documents(pages)
    return docs

###############################################################################
def build_or_load_index(pdf_path, index_path, embed_model_name = "intfloat/multilingual-e5-base", rebuild = False):
    embeddings = HuggingFaceEmbeddings(model_name=embed_model_name, model_kwargs={"device": "cpu"}, encode_kwargs={"normalize_embeddings": True})
    if index_path.exists() and not rebuild:
        return FAISS.load_local(str(index_path), embeddings, allow_dangerous_deserialization=True)

    docs = load_and_split(pdf_path)

    # Префикс E5 для documents
    for d in docs:
        d.page_content = "passage: " + d.page_content

    vs = FAISS.from_documents(docs, embeddings)
    index_path.mkdir(parents=True, exist_ok=True)
    vs.save_local(str(index_path))
    logger.info(f"Индекс сохранён в {index_path}")
    return vs

###############################################################################
# поиск релевантных фрагментов в индексе FAISS СиПР
def retrieve_sipr(vs, query, k = 5):
    prefixed_query = "query: " + query
    docs = vs.similarity_search(prefixed_query, k=k)
    if not docs:
        return "В документе СиПР 2025-2030 не найдено релевантных фрагментов"
    chunks = []
    for i, d in enumerate(docs, 1):
        page = d.metadata.get("page", "?")
        text = d.page_content
        text = text.replace("passage: ", "", 1).strip()
        chunks.append(f"[Фрагмент {i}, стр. {page}]\n{text}")

    return "\n\n".join(chunks)