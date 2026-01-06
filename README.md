# battery-research-data-assistant

## Data
I have used the "paper-abstracts" dataset from "batterydata" on Hugging Face datasets repository for this project. It can be found here (https://huggingface.co/datasets/batterydata/paper-abstracts).

It contains abstracts of reserach papers related to battery technology as well as many non-battery-related papers. However for the purpose of this project only battery-related abstracts were used.

**Citation**-
@article{huang2022batterybert,
  title={BatteryBERT: A Pretrained Language Model for Battery Database Enhancement},
  author={Huang, Shu and Cole, Jacqueline M},
  journal={J. Chem. Inf. Model.},
  year={2022},
  doi={10.1021/acs.jcim.2c00035},
  url={DOI:10.1021/acs.jcim.2c00035},
  pages={DOI: 10.1021/acs.jcim.2c00035},
  publisher={ACS Publications}
}

## Data Ingestion and Storage
data.py file contains code for the entire data ingestion process. It downloads the data from Hugging Face datasets repository, filters out any non-battery related data, create embeddings for each scientific paper abstract and saves it alongwith other metadata in a vector database (ChromaDB).
