# ClinLLM
This repository provides the official implementation for the paper *ClinLLM: A Time-Aware Approach for Integrating Structured EHRs encoders with Large Language Models*.

In this paper, we introduce **ClinLLM**, a time-aware large language model (LLM) framework that integrates structured EHR signals through soft prompt tuning. The goal of this work is to address issues that arise when applying LLM methods to EHR data; namely,  (i) the limitations of free text in representing the temporal structure of medical codes, and (ii) the difficulty of modeling patient-to-patient similarity patterns that underpin the success of traditional EHR models.

## Installation
To set up the dependencies, simply install the packages listed in `requirements.txt`:
```bash
pip install -r requirements.txt
```