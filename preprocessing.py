import time
startTime = time.time()
import pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

csv1 = pd.read_csv("Desktop/NLP/archive/articles1.csv")
csv2 = pd.read_csv("Desktop/NLP/archive/articles2.csv")
csv3 = pd.read_csv("Desktop/NLP/archive/articles3.csv")

data = csv1 + csv2 + csv3
data = data[pd.isna(data['content']) == False]
contents = data['content']

stop_words = ['a', 'about', 'after', 'all', 'also', 'always']
model = TfidfVectorizer(stop_words = stop_words)
tfidf = model.fit_transform(contents)

pickle.dump(data, open("data.pickle", "wb"))
pickle.dump(tfidf, open("tfidf.pickle", "wb"))
pickle.dump(model, open("model.pickle", "wb"))
print('preprocessing ended and it took {0} seconds'.format(time.time() - startTime))