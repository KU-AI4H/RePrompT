from pyhealth.datasets import split_by_patient, get_dataloader
import pickle
from pyhealth.datasets import get_dataloader
from CustomTrainer import Trainer
from RETAIN_as_Softprompt_sequential import RETAIN

for seed in [42, 24, 12]:
    for task in ['readmission', 'mortality']:
        if task == 'readmission':
            with open("processed_data_mimicIII/dataset_rad.pkl", "rb") as f:
                mimic3sample = pickle.load(f)
        else:
            with open("processed_data_mimicIII/dataset_mrt.pkl", "rb") as f:
                mimic3sample = pickle.load(f)
        train_ds, val_ds, test_ds = split_by_patient(mimic3sample, [0.7, 0.0, 0.3], seed=42)
        train_loader = get_dataloader(train_ds, batch_size=8, shuffle=True)
        val_loader = get_dataloader(val_ds, batch_size=8, shuffle=False)
        test_loader = get_dataloader(test_ds, batch_size=8, shuffle=False)

        model = RETAIN(
                feature_keys=['conditions', 'procedures','drugs_hist'],
                label_key="label",
                embedding_dim=256,
                dataset=mimic3sample,
                mode='binary'
            )
        print('Information', seed, task)
        trainer = Trainer(model=model)
        trainer.train(
            train_dataloader=train_loader,
            val_dataloader=test_loader,
            epochs=20,
            monitor="pr_auc",
            optimizer_params={"lr": 1e-4},  # Using learning rate of 5e-5
            load_best_model_at_last = False,
            the_text=f'{seed}_{task}_MIMICIII'

    )