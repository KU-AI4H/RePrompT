from __future__ import annotations

from typing import Literal, get_args
from pathlib import Path
from collections import defaultdict
from openai import OpenAI
import os
from pyhealth.datasets.sample_dataset import SampleEHRDataset
from torch.utils.data import Subset

class NShotBinaryClassifier:
    """
    Class for performing one-shot binary classification using an LLM.
    
    Attributes:
        SupportedTasks: The supported binary classification tasks
                        ("mortality_prediction", "readmission_prediction").
    """
    SupportedTask = Literal["mortality_prediction", "readmission_prediction"]

    task_features = {
        "mortality_prediction": {
            "conditions_description": "The patient's conditions",
            "procedures_description": "The patient's procedures",
            "drugs_description": "The patient's drug history"
        },
        "readmission_prediction": {
            "conditions_description": "The patient's conditions",
            "procedures_description": "The patient's procedures",
            "drugs_hist_description": "The patient's drug history"
        },
    }

    def __init__(
        self,
        task: SupportedTask,
        n_shots: int = 0,
        model_id: str = "gpt-5"
    ):
        supported_tasks = get_args(self.SupportedTask)
        if task not in supported_tasks:
            raise ValueError(
                f"Task '{task}' is not supported. Valid tasks: {supported_tasks}"
            )
        
        self.task = task
        self.prompt = self._load_prompt_template(self.task)
        self.n_shots = n_shots
        self.model_id = model_id

        openai_api_key = os.getenv("OPENAI_API_KEY")

        if openai_api_key is None:
            raise ValueError("OpenAI API key environment variable is not set.")

        self.llm_client = OpenAI(
            api_key=openai_api_key
        )

    @staticmethod
    def _load_prompt_template(task: str) -> str:
        """
        Loads the prompt template for the selected task.

        Returns:
            The prompt template, loaded from `./prompts`, for the predictor's task.
        """
        template_path = Path("prompts") / f"{task}_template.txt"
        
        with open(template_path, "r") as file:
            return file.read()

    @staticmethod 
    def _format_patient_info(
        visits: list[dict],
        features: dict[str, str]
    ) -> str:
        """
        Format a patient's info for use in an LLM prompt.

        Args:
            visits: The patient's visits to format.
            features: The features to show from the visits.
        Returns:
            The features in a human-readable string format.
        """
        feature_dict = defaultdict(list)
        
        for visit in visits:
            for key, value in visit.items():
                feature_dict[key].append(value)

        patient_info = (
            f"The patient had {len(visits)} visits that occurred at "
            f"{", ".join([str(i) for i in range(len(visits))])}.\n"
            "Details of the features for each visit are as follows:\n"
        )

        for key, value in features.items():
            patient_info += f"- {value}: {feature_dict[key][0]}\n"

        patient_info += (
            "\nRESPONSE:\n"
            f"{visits[-1]['label']}\n\n"
        )

        return patient_info
        
    def _format_prompt(
        self,
        visits: list[dict],
        example_visits: list[list[dict]] | None = None
    ) -> str:
        """
        Format a prompt by interpolating patient visit history information.

        Args:
            visits: A patient's visit history.
            example_visits: The visits to use as examples for n-shot prompting.
        Returns:
            The formatted prompt, containing information about the patient's visit
            history.
        """
        selected_features = self.task_features.get(self.task)

        patient_info = self._format_patient_info(visits, selected_features)

        example_patient_information = ""
        if example_visits is not None and len(example_visits) > 0:
            example_patient_information = "Here is an example of input information:\n"
            for i, patient_visits in enumerate(example_visits):
                example_patient_information += f"Example #{i + 1}\n"
                example_patient_information += self._format_patient_info(
                    patient_visits, selected_features
                )

        prompt = self.prompt.format(
            patient_information=patient_info,
            example_patient_information=example_patient_information
        )

        return prompt
    
    def _llm_inference(self, prompt: str) -> str:
        """
        Obtain an LLM completion.

        Args:
            prompt: The prompt to use for LLM inference.
        Returns:
            out: The LLM's text completion.
        """
        return self.llm_client.responses.create(
            model=self.model_id,
            input=prompt
        ).output[0].content[0].text
    
    @staticmethod
    def _aggregate_patient_visits(data: Subset | SampleEHRDataset) -> list:
        """
        Aggregate all visits by patient ID.

        Args:
            data: The visits to aggregate.
        Returns:
            A list of visits, each one containing all visits belonging to one patient.
        """
        if isinstance(data, Subset):
            data = SampleEHRDataset(data)
        patient2index = data.patient_to_index
        aggregated_visits = []
        for patient in patient2index.keys():
            patient_visit_indices = patient2index[patient]
            patient_visits = [data[index] for index in patient_visit_indices]
            aggregated_visits.append(patient_visits)
        return aggregated_visits

    def forward(
        self,
        patient_data: Subset | SampleEHRDataset,
        sample_data: Subset | SampleEHRDataset,
    ):
        """
        Generate a list of binary predictions using n-shot prompting.

        Args:
            patient_data: The data for which to perform class label predictions.
            sample_data: The data to use for n-shot examples.
        Returns:
            A list of values between 0 and 1 denoting the likelihood of each sample
            belonging to the 'true' class.
        """
        sample_visits = self._aggregate_patient_visits(sample_data)
        sample_data = sample_visits[:self.n_shots]

        visits_by_patient = self._aggregate_patient_visits(patient_data)
        prompts = [
            self._format_prompt(patient_visits, sample_data)
            for patient_visits in visits_by_patient
        ]

        # Obtain a predicted label for each patient
        predicted_labels = [self._llm_inference(prompt) for prompt in prompts]

        return predicted_labels

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    