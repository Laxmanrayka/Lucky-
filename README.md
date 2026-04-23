# Fake News Detection using BERT

Detect fake/real news articles using BERT transformer models, with full explainability (SHAP) and an end-to-end ML workflow.  
Colab, GPU, and local Python compatible.

---

## 🚀 Features

- Fine-tuning BERT for binary news classification (Fake vs Real)
- Simple and robust data pipeline (pandas, scikit-learn, HuggingFace Datasets)
- Model explainability using [SHAP](https://shap.readthedocs.io/) (why is news predicted Fake/Real)
- Saves trained model for easy re-use
- Supports Jupyter/Colab **and** local Python

---

## 🗂️ Dataset

This project expects [Fake and Real News Dataset](https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset) from Kaggle.

**Required files:**
- `True.csv` &rarr; real news (`title`, `text` columns)
- `Fake.csv` &rarr; fake news (`title`, `text` columns)

Place these files in your working directory.

---

## 🛠 Installation

**1. Python version**  
```bash
python 3.8+ recommended (3.7-3.11 supported)
```

**2. Install required packages:**
```bash
pip install transformers datasets torch pandas scikit-learn shap seaborn matplotlib
```
> Note: Ignore any error about `sha` package – correct name is `shap`.

---

## ⚡ Quickstart

**Run on Google Colab:**  
> Upload `True.csv` and `Fake.csv` before running.

```python
# 1. Load and preprocess data (adjust file paths as needed)
import pandas as pd

true = pd.read_csv('True.csv', engine='python', on_bad_lines='skip')
fake = pd.read_csv('Fake.csv', engine='python', on_bad_lines='skip')
true['label'] = 1
fake['label'] = 0
df = pd.concat([true, fake], ignore_index=True)
df['news'] = df['title'] + " " + df['text']
df = df.sample(frac=1, random_state=42).reset_index(drop=True)[['news', 'label']]

# 2. Train/test split
from sklearn.model_selection import train_test_split
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
```

**Model training and evaluation:**  
Read and follow the `notebooks/FakeNews-BERT.ipynb` for full code and walkthrough, or see [Usage Example](#-usage-example).

---

## 📂 Usage Example

*Train and predict with custom news inputs:*

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, pipeline
from datasets import Dataset

# Prepare datasets
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

def tokenize_function(examples):
    return tokenizer(examples['news'], padding='max_length', truncation=True, max_length=512)

tokenized_train = train_dataset.map(tokenize_function, batched=True).remove_columns(['news'])
tokenized_test = test_dataset.map(tokenize_function, batched=True).remove_columns(['news'])
tokenized_train.set_format("torch")
tokenized_test.set_format("torch")

model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    report_to="none"
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test
)
trainer.train()
# Save model
model.save_pretrained('./fake_news_bert')
tokenizer.save_pretrained('./fake_news_bert')

# Predict on new data
classifier = pipeline("text-classification", model='./fake_news_bert', tokenizer='./fake_news_bert')
print(classifier("Scientists discover new planet with water and life signs"))
```

---

## 📊 Visualization & Interpretability

- **Confusion matrix** and **classification report** in `matplotlib` plots
- **Model explainability** with SHAP:
    ```python
    import shap
    explainer = shap.Explainer(classifier)
    shap_values = explainer(["Donald Trump announces he is actually an alien from Mars"])
    shap.plots.text(shap_values[0])
    ```
- Results explain **which words contributed most** to the label (Fake/Real).
- See `notebooks/` for all metrics, plots, and explainability outputs.

---

## 🖼️ Example Output

![Confusion Matrix](confusion_matrix_example.png)  
*Confusion matrix for test set predictions.*

![SHAP Example](shap_example.png)  
*SHAP interpretability plot showing token contributions.*

---

## 🤝 Contributing

1. Fork this repo
2. Make your changes with clear commit messages
3. Open a pull request!

---

## 📑 License

MIT License

---

## 🙏 Credits

- [HuggingFace Transformers](https://huggingface.co/docs)
- [scikit-learn](https://scikit-learn.org/)
- [SHAP](https://shap.readthedocs.io/en/latest/)
- [Fake and Real News Dataset (Kaggle)](https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset)

---

Jai Mama Dhani 🚩  
