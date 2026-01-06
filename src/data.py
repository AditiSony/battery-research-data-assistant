import os
import shutil
import time

from datasets import load_dataset
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from settings import BATCH_SIZE, DEVICE, EMBED_MODEL, VECTOR_DB_DIR


def data_ingestion():
    if os.path.exists(VECTOR_DB_DIR):
        print("Cleaning up old database")
        shutil.rmtree(VECTOR_DB_DIR)

    # read data from huggingface datasets and keep only battery-related data
    dataset = load_dataset("batterydata/paper-abstracts", split="train")
    battery_data = dataset.filter(lambda x: x["label"] == "battery")
    battery_data = battery_data.select(range(10))

    # create vector embeddings for all docs
    encode_kwargs = {"normalize_embeddings": True}
    query_encode_kwargs = {
        "prompt": "Represent this sentence for searching relevant passages: "
    }

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": DEVICE},
        encode_kwargs=encode_kwargs,
        query_encode_kwargs=query_encode_kwargs,
    )

    # Store embeddings in Vector Database (ChromaDB)
    print(f"Starting Data Ingestion (Batch Size: {BATCH_SIZE})...")

    # Consider each row/ abstract as a single document
    first_batch = battery_data.select(range(0, BATCH_SIZE))
    docs = [
        Document(
            page_content=x["abstract"],
            metadata={
                "source": "batterydata/paper-abstracts",
                "category": x["label"],
                "data_type": "scientific-abstract",
            },
        )
        for x in first_batch
    ]

    vector_db = Chroma.from_documents(
        documents=docs, embedding=embeddings, persist_directory=VECTOR_DB_DIR
    )
    print(f"Progress: {BATCH_SIZE}/{len(battery_data)} documents indexed...")

    if len(battery_data) > BATCH_SIZE:
        for i in range(BATCH_SIZE, len(battery_data), BATCH_SIZE):
            end_idx = min(i + BATCH_SIZE, len(battery_data))
            batch = battery_data.select(range(i, end_idx))

            batch_docs = [
                Document(
                    page_content=x["abstract"],
                    metadata={
                        "source": "batterydata/paper-abstracts",
                        "category": x["label"],
                        "data_type": "scientific-abstract",
                    },
                )
                for x in batch
            ]

            vector_db.add_documents(batch_docs)
            print(f"Progress: {end_idx}/{len(battery_data)} documents indexed...")


if __name__ == "__main__":
    start_time = time.time()
    data_ingestion()
    print(f"Total Time Taken: {(time.time() - start_time) / 60:.2f} minutes")
