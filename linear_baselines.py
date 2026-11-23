import json
from collections import Counter
from datasets import Dataset, load_dataset
from datasets.config import HF_DATASETS_CACHE
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, f1_score
import sklearn

from prepare_banking77 import DATASET_ID, DATASET_REVISION

cached = sorted(HF_DATASETS_CACHE.glob(
    f"PolyAI___banking77/**/{DATASET_REVISION}/banking77-train.arrow"
))
if cached:
    raw = Dataset.from_file(str(cached[0]), in_memory=True)
else:
    raw = load_dataset(DATASET_ID, revision=DATASET_REVISION, split="train")
seen = set()
keep = []
for i, text in enumerate(raw["text"]):
    key = " ".join(text.lower().split())
    if key not in seen:
        keep.append(i)
        seen.add(key)
partitions = raw.select(keep).train_test_split(test_size=.10, stratify_by_column="label", seed=42)
train, val = partitions["train"], partitions["test"]
names = raw.features["label"].names
model = make_pipeline(TfidfVectorizer(ngram_range=(1, 2)), LogisticRegression(C=1, max_iter=1000))
model.fit(list(train["text"]), list(train["label"]))
majority = Counter(train["label"]).most_common(1)[0][0]
scores = {}
for name, predictions in (("majority", [majority] * len(val)), ("tfidf_logistic", model.predict(list(val["text"])))):
    scores[name] = {"accuracy": accuracy_score(list(val["label"]), predictions),
                    "macro_f1": f1_score(list(val["label"]), predictions, labels=list(range(77)), average="macro", zero_division=0)}
results = {"train":len(train), "validation":len(val), "seed":42, "sklearn":sklearn.__version__,
           "tfidf_ngrams":[1,2], "logistic_C":1, "majority_label":names[majority], **scores}
print(json.dumps(results, indent=2))
