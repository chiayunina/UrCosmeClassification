import os
import numpy as np

from tqdm import tqdm

from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline

# == Load data ==
print("Loading dataset...")
files = os.listdir()
D = []

# Label:
# - dry: 0
# - normal: 1
# - oil: 2
label = np.array([])

print(len(D), " documents, ", len(label), " labels.")

# == Chinese Segmentation ==
for 

# == Vector Transformation ==
print("Extracting features from the dataset...")
vectorizer = Pipeline([
                       ('vect', HashingVectorizer(n_features=(2 ** 21), non_negative=True, lowercase=False)),
                       ('tfidf', TfidfTransformer(norm='l2')),
                       ])

if __name__ == '__main__':
    vectorizer.fit()