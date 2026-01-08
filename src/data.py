import os
import random
import shutil
import time
import warnings

from datasets import load_dataset
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from settings import BATCH_SIZE, DEVICE, EMBED_MODEL, VECTOR_DB_DIR


def data_ingestion():
    if os.path.exists(VECTOR_DB_DIR):
        print(f"Database found at {VECTOR_DB_DIR}. Skipping ingestion.")
        return

    print("No database found. Starting fresh ingestion...")

    # read data from huggingface datasets and keep only battery-related data
    dataset = load_dataset("batterydata/paper-abstracts", split="train")
    battery_data = dataset.filter(lambda x: x["label"] == "battery")

    # Add fictitious "journal" column
    journals = ["Journal A", "Journal B", "Journal C", "Journal D"]
    battery_data = battery_data.map(lambda x: {"journal": random.choice(journals)})

    batch_size = BATCH_SIZE
    if len(battery_data) < batch_size:
        warnings.warn(
            f"Battery Dataset size ({len(battery_data)}) is smaller than selected batch size {batch_size}. Selecting all battery data for ingestion."
        )
        batch_size = len(battery_data)

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
    print(f"Starting Data Ingestion (Batch Size: {batch_size})...")

    # Consider each row/ abstract as a single document
    first_batch = battery_data.select(range(0, batch_size))
    docs = [
        Document(
            page_content=x["abstract"],
            metadata={
                "source": x["journal"],
                "category": x["label"],
                "data_type": "scientific-abstract",
            },
        )
        for x in first_batch
    ]

    vector_db = Chroma.from_documents(
        documents=docs, embedding=embeddings, persist_directory=VECTOR_DB_DIR
    )
    print(f"Progress: {batch_size}/{len(battery_data)} documents indexed...")

    if len(battery_data) > batch_size:
        for i in range(batch_size, len(battery_data), batch_size):
            end_idx = min(i + batch_size, len(battery_data))
            batch = battery_data.select(range(i, end_idx))

            batch_docs = [
                Document(
                    page_content=x["abstract"],
                    metadata={
                        "source": x["journal"],
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
