import numpy as np
import jieba
from seg_zhtw import seg
from tqdm import tqdm

from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer

# Load data
print("Loading dataset...")
D = []

# Label:
# - dry: 0
# - normal: 1
# - oil: 2
label = np.array([])

print(len(D), " documents, ", len(label), " labels.")

# == Vector Transformation ==
print("Extracting features from the dataset...")
vectorizer = Pipeline([
                       ('vect', HashingVectorizer(n_features=(2 ** 21), non_negative=True)),
                       ('tfidf', TfidfTransformer(norm='l2')),
                       ])