#from pyhealth.datasets import MIMIC3Dataset
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import pickle
from CustomizedPyhealth.mimic3 import MIMIC3Dataset
import pickle
import requests
import json
import re
import os

from dotenv import load_dotenv
load_dotenv()

all_notes = []
all_procedures = []
all_diagnosis = []
all_drugs_1 = []
from pyhealth.medcode import InnerMap

icd9dg = InnerMap.load("ICD9CM")
icd9pr = InnerMap.load("ICD9PROC")
ndc9dr = InnerMap.load("NDC")


def deepseek_inference(prompt):
    api_key = os.getenv("DEEPSEEK_API_KEY")

    if api_key is None:
        raise ValueError("DeepSeek API key environment variable is not set.")

    url = "https://api.deepseek.com/chat/completions"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    data = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": "You are an expert AI agent judge in medical science."},
            {"role": "user", "content": f"{prompt}"}
        ],
        "temperature": 0.2
    }

    response = requests.post(url, headers=headers, json=data)
    #print('response', response)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]



def medical_codes_description_builder(list_medical_codes):
    description = ''
    for n_visit in range(0, len(list_medical_codes)):
        medical_code_visit = list_medical_codes[n_visit]
        #print('nedical_code_visit', medical_code_visit)
        if len(medical_code_visit) == 0:
            description += f' **** Visit {n_visit} *****: specific type of medical code not available'
        else:
            #print('medical code information:', medical_code_visit)
            medical_code_information = ', '.join(medical_code_visit)
            description += f' **** Visit {n_visit} *****: [{medical_code_information}]'
    return description
def prompt_builder(diagnosis_codes, procedure_codes, medications, discharge_notes):

    prompt = f"""
    You are a clinical summarization assistant. Given structured medical data across multiple visits 
    and discharge documentation, generate a concise but detailed patient summary. 
    The summary must not exceed 700 tokens. Write in a professional, clinical style suitable for 
    healthcare documentation. Explicitly address comorbidities, longitudinal trends across visits, 
    and mortality considerations. Note that Visit 1 is the first visits and Visit N is the last visit and you should mainly focus on the last visit for its importance.

    Patient data for summarization (spanning multiple visits):

    - **** ICD-9 Diagnosis Codes ****: [{medical_codes_description_builder(diagnosis_codes)}]
    - **** ICD-9 Procedure Codes ****: [{medical_codes_description_builder(procedure_codes)}]
    - **** Medications NDC codes ****: [{medical_codes_description_builder(medications)}]
    - **** Discharge Notes ****: [{medical_codes_description_builder(discharge_notes)}]
    Please generate a structured output with the following sections:

    1. Longitudinal Patient Summary (≤500 tokens): Cohesive narrative integrating diagnoses, 
       procedures, medications, and discharge details across multiple visits. Highlight disease 
       progression or recurring conditions.

    2. Risk & Mortality Considerations (≤100 tokens): Emphasize comorbidities, treatment history, 
       complications, and longitudinal risk factors that may influence mortality.

    3. Overall Clinical Impression (≤100 tokens): Concise synthesis of the patient’s status, 
       prognosis, and key follow-up considerations across visits.
    """
    #print('prompt_per_visit', prompt)
    #

    return prompt
def process_single_data(lookup, code, type = 'ICD 9 code:'):
    try:
        #print('code', code)
        description = lookup.lookup(code)
        #print('description', description)
    except:
        return None
    details = type + str(code) + ' Description: ' + description
    return details
'''
def process_sequence_data(lookup, sequence, type = 'ICD 9 code:'):
    all_sequence = []
    for item in sequence:
        item_desc = process_single_data(item, lookup, type)
        if item_desc is not None:
            all_sequence.append(item_desc)
    if len(all_sequence)> 0:
        return all_sequence
    else:
        return ['na']
'''
n_data_point = 0
def process_sequence_data(lookup, sequence, type='ICD 9 code:'):
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(lambda item: process_single_data(lookup, item, type), sequence))

    all_sequence = [res for res in results if res is not None]
    return all_sequence if all_sequence else ['na']

def get_dictionary():
    with open("/home/a053h213/PycharmProjects/EHRLLMGraph/MortalityPrediction/EHR_SOFT_PROMPTS/dataset.pkl", "rb") as f:
        mimic3sample = pickle.load(f)
    llm_dict = {}
    for item in mimic3sample:
        llm_dict[str(item['visit_id']) + '_' + str(item['patient_id'])] = item['llm_output']
    return llm_dict

def sequential_drug_recommendation_mimic(patient):
    global n_data_point
    sequential_conditions = []
    sequential_procedures = []
    sequential_drugs = []
    sequential_notes = []
    sequential_conditions_description = []
    sequential_procedures_description = []
    sequential_drugs_description = []
    samples = []
    sequential_llm_information = []
    l_dict = get_dictionary()
    for i in range(len(patient) - 1):
        #print(i)
        visit = patient[i]
        next_visit = patient[i+1]
        time_diff = (next_visit.encounter_time - visit.encounter_time).days
        time_window = 15
        readmission_label = 1 if time_diff < time_window else 0
        conditions = visit.get_code_list(table="DIAGNOSES_ICD")
        procedures = visit.get_code_list(table="PROCEDURES_ICD")
        drugs = visit.get_code_list(table="PRESCRIPTIONS")
        sequential_conditions_description.append(process_sequence_data(icd9dg, conditions, 'ICD 9 diagnosis code:'))
        sequential_procedures_description.append(process_sequence_data(icd9pr, procedures, 'ICD 9 procedure code:'))
        sequential_drugs_description.append(process_sequence_data(ndc9dr, drugs, 'NDC drug code:'))

        if next_visit.discharge_status not in [0, 1]:
            mortality_label = 0
        else:
            mortality_label = int(next_visit.discharge_status)
        notes = visit.get_code_list(table="NOTEEVENTS")
        all_notes.append(notes)
        if len(drugs) == 0:
            sequential_drugs.append(['N/A'])
        else:
            sequential_drugs.append(drugs)
        if len(conditions) == 0:
            sequential_conditions.append(['N/A'])
        else:
            sequential_conditions.append(conditions)
        if len(procedures) == 0:
            sequential_procedures.append(['N/A'])
        else:
            sequential_procedures.append(procedures)
        if len(notes) == 0:
            sequential_notes.append(['N/A'])
        else:
            sequential_notes.append(notes)
        n_data_point += 1
        original_prompt = prompt_builder(sequential_conditions_description, sequential_procedures_description, sequential_drugs_description, sequential_notes)

        '''try:
            output = deepseek_inference(original_prompt)
        except:
            output = 'N/A'
        '''
        sequential_llm_information = l_dict[str(visit.visit_id) + '_' + str(patient.patient_id)]
        #print('seq_llm_info', len(sequential_llm_information), len([item for item in sequential_drugs if not ['N/A'] == item]))
        samples.append(
            {
                "visit_id": visit.visit_id,
                "patient_id": patient.patient_id,
                "notes": sequential_notes.copy(),
                "conditions": sequential_conditions.copy(),
                "procedures": sequential_procedures.copy(),
                "drugs_hist": sequential_drugs.copy(),
                "llm_output": sequential_llm_information.copy(),
                "conditions_description": sequential_conditions_description.copy(),
                "procedures_description": sequential_procedures_description.copy(),
                "drugs_description": sequential_drugs_description.copy(),
                "label": readmission_label
            }

        )
    return samples



print('this line achieved')

dataset1 = MIMIC3Dataset(
    root="/home/a053h213/PycharmProjects/PrescriptionRecommendation/Datasets/MIMICIII/data",
    tables=[ "NOTEEVENTS", "DIAGNOSES_ICD", "PROCEDURES_ICD", "PRESCRIPTIONS"],
    #dev=True
    #refresh_cache=True,
    #code_mapping={"NDC": ("ATC", {"target_kwargs": {"level": 3}}),
                  #"ICD9CM": {"target_kwargs": {"level": 3}},
                  #"ICD9PROC": {"target_kwargs": {"level": 3}}
    #              })
)

dataset = dataset1.set_task(task_fn=sequential_drug_recommendation_mimic)
with open("/home/a053h213/PycharmProjects/EHRLLMGraph/MortalityPrediction/EHR_SOFT_PROMPTS/dataset_rad.pkl", "wb") as f:
    pickle.dump(dataset, f)
print(n_data_point)


