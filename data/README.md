# Data

This project uses finance QA data derived from FinQA/TAT-QA through the Hugging Face dataset `hellotayssir/FinQA_TAT-QA_financial_finetuning_dataset` (CC BY 4.0 as documented by the dataset card). The preparation script downloads the public data, normalizes it into prompt/completion JSONL, deduplicates, creates validation from the provided training split, and preserves the provided test split for held-out evaluation.


# Financial data

Put source PDF/TXT/MD documents under `data/raw/company_reports/` for RAG.

Example:
```text
data/raw/company_reports/
├── reliance_annual_report_2024.pdf
├── tcs_annual_report_2024.pdf
├── infosys_notes.txt
└── market_research.md
```

These documents are used by the RAG ingestion pipeline, not automatically by QLoRA training.
