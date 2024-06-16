import torch
from torch import nn
import lightning as L
import torch.nn.functional as F
from torchmetrics import Accuracy
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
from torch.utils.data import DataLoader, Dataset, random_split, TensorDataset
import pandas as pd
from lightning.pytorch.loggers import TensorBoardLogger
from datasets import load_dataset, Dataset as DS
import matplotlib.pyplot as plt
import numpy as np
from umap import UMAP
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import RandomOverSampler, SMOTE
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, f1_score
import evaluate
from transformers.integrations import TensorBoardCallback
from accelerate import Accelerator
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

accelerator = Accelerator()

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = accelerator.device
MODEL = "microsoft/deberta-large"
# MODEL = "roberta-large"

def compute_metrics(eval_preds):
    accuracy_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)
    f1 = f1_metric.compute(predictions=predictions, references=labels, average="weighted")
    return {"accuracy": accuracy["accuracy"], "f1": f1["f1"]}


tokenizer = AutoTokenizer.from_pretrained(MODEL)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


revies_ds = load_dataset("csv", data_files="data/train_data.csv", sep=",")
labels = revies_ds['train']['rating']
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(labels), y=labels)
CLS_WEIGHTS = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)

review_encoded = revies_ds.map(lambda ds: tokenizer(ds['review'], truncation=True), batched=True)
review_encoded=review_encoded.remove_columns(['review'])
review_encoded=review_encoded.rename_column('rating', 'labels')
review_encoded=review_encoded['train'].train_test_split(test_size=0.2)
# review_encoded['train'] = resample(review_encoded, 'train')

num_labels = 5
model = (AutoModelForSequenceClassification
         .from_pretrained(MODEL, num_labels=num_labels)
         .to(DEVICE))
batch_size = 1
accumulation_steps = 4  # Accumulate gradients over 4 steps
logging_steps = len(review_encoded["train"]) // batch_size
training_args = TrainingArguments('bert-finetune',
                                  num_train_epochs=1,
                                  learning_rate=2e-5,
                                  per_device_train_batch_size=batch_size,
                                  per_device_eval_batch_size=batch_size,
                                #   gradient_accumulation_steps=accumulation_steps,
                                  weight_decay=0.01,
                                  evaluation_strategy="epoch",
                                  disable_tqdm=False,
                                  logging_steps=logging_steps,
                                  push_to_hub=False,
                                  gradient_accumulation_steps=8,      # Accumulate gradients to fit large batch size
                                  dataloader_num_workers=4,   
                                  fp16=True,
                                  log_level="error")
class WeightedTrainer(Trainer):
    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.
        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs)
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            loss = self.label_smoother(outputs, labels)
        else:
            logits = outputs['logits']
            criterion = torch.nn.CrossEntropyLoss(weight=self.class_weights)
            loss = criterion(logits, inputs['labels'])

        return (loss, outputs) if return_outputs else loss
trainer = WeightedTrainer(model=model, args=training_args,
                  compute_metrics=compute_metrics,
                  train_dataset=review_encoded['train'],
                  eval_dataset=review_encoded['test'],
                  tokenizer=tokenizer,
                  data_collator=data_collator,
                  class_weights=CLS_WEIGHTS,
                  )
trainer.train();
trainer.save_model("final_model")

test_ds = load_dataset("csv", data_files="data/test_data.csv", sep=",", names=['review'])
test_encoded = test_ds.map(lambda ds: tokenizer(ds['review'], truncation=True), batched=True)
test_encoded = test_encoded.remove_columns(['review'])
test_encoded = test_encoded['train']
predictions = trainer.predict(test_encoded)
predicted_labels = np.argmax(predictions.predictions, axis=-1)
pd.DataFrame(predicted_labels, columns=['Values']).to_csv('large_result.csv', index=False, header=False)

predictions = trainer.predict(review_encoded['test'])
true_labels = predictions.label_ids
predicted_labels = np.argmax(predictions.predictions, axis=-1)
conf_matrix = confusion_matrix(true_labels, predicted_labels)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=[0, 1, 2, 3, 4])
disp.plot(cmap=plt.cm.Blues)
plt.title('DeBerta Confusion Matrix')
plt.savefig('DeBERT_confusion_matrix.png')
plt.show()