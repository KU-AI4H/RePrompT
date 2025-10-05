from __future__ import annotations

from typing import Literal, get_args
from pathlib import Path
from collections import defaultdict
import os
from pyhealth.datasets.sample_dataset import SampleEHRDataset
from torch.utils.data import Subset
from inference import OpenAIInference
from tqdm import tqdm

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
            "drugs_description": "The patient's drug history"
        },
    }

    def __init__(
        self,
        task: SupportedTask,
        n_shots: int = 0,
        model_id: str = "gpt-5",
        log_path: str | Path = None
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
        self.log_path = Path(log_path) if log_path else None

        openai_api_key = os.getenv("OPENAI_API_KEY")

        if openai_api_key is None:
            raise ValueError("OpenAI API key environment variable is not set.")

        self.llm_client = OpenAIInference(api_key=openai_api_key, model=model_id)

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
        features: dict[str, str],
        include_label: bool = False
    ) -> str:
        """
        Format a patient's info for use in an LLM prompt.

        Args:
            visits: The patient's visits to format.
            features: The features to show from the visits.
        Returns:
            The features in a human-readable string format.
        """
        feature_dict = {}
        
        for key, value in visits.items():
            feature_dict[key] = value

        patient_info = (
            f"The patient had {len(visits['notes'])} visits that occurred at "
            f"{", ".join([str(i) for i in range(len(visits['notes']))])}.\n"
            "Details of the features for each visit are as follows:\n"
        )

        for key, value in features.items():
            patient_info += f"- {value}: {feature_dict[key]}\n"

        if include_label:
            patient_info += (
                "\nRESPONSE:\n"
                f"{visits['label']}\n\n"
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
                    patient_visits, selected_features, include_label=True
                )

        prompt = self.prompt.format(
            patient_information=patient_info,
            example_patient_information=example_patient_information
        )

        return prompt

    def forward(
        self,
        patient_data: Subset | SampleEHRDataset,
        sample_data: Subset | SampleEHRDataset | None = None,
        start_index: int | None = None,
        label: str = "label"
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
        prompts = [
            self._format_prompt(visit_samples, sample_data)
            for visit_samples in patient_data
        ]

        # Obtain a predicted label for each patient
        predicted_labels = []

        # Optionally slice prompts and visits
        if start_index is not None:
            prompts = prompts[start_index:]
            patient_data = [patient for i, patient in enumerate(patient_data) if i >= start_index]

        for prompt, visits in tqdm(
            zip(prompts, patient_data),
            total=len(prompts),
            unit="patient"
        ):
            predicted_label = self.llm_client.generate(prompt)
            predicted_labels.append(predicted_label)

            true_label = visits[label]

            if self.log_path is not None:
                with open(self.log_path, "a") as log_file:
                    log_file.write(
                        f"{visits['patient_id']},"
                        f"{visits['visit_id']},"
                        f"{predicted_label},"
                        f"{true_label}\n"
                    )

        return predicted_labels, prompts

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    