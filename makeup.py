import os
import numpy as np
from segment import segmenter, del_stops
from tqdm import tqdm

from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

# == Load data ==
print("Loading dataset...")
files = os.listdir()
D = []

# Label:
# - dry: 0
# - normal: 1
# - oil: 2
y = np.array([])
print(len(D), " documents, ", len(y), " labels.")

# == Chinese Segmentation ==
X = np.array([del_stops(segmenter(d)) for d in D])

# == Split dataset ==
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# == Vector Transformation ==
print("Extracting features from the dataset...")
vectorizer = Pipeline([
                       ('vect', HashingVectorizer(n_features=(2 ** 21), non_negative=True, lowercase=False)),
                       ('tfidf', TfidfTransformer(norm='l2')),
                       ])

if __name__ == '__main__':
    vectorizer.fit()