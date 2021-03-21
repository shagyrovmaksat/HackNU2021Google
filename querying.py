import time
startTime = time.time()
import pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

data = pickle.load(open("data.pickle", "rb"))
tfidf = pickle.load(open("tfidf.pickle", "rb"))
model = pickle.load(open("model.pickle", "rb"))

def search(request):
    request_transform = model.transform([request])
    similarity = np.dot(request_transform, np.transpose(tfidf))
    x = np.array(similarity.toarray()[0])
    indices = np.argsort(x)[-5:][::-1]
    return indices

startOfRequest = 0
def printResult(indices):
    print('Best Results :')
    for i in range(0,len(indices)):
        print( '\n{0}) id = {1}; content - {2}'.format(i+1, indices[i], data['content'].loc[indices[i]]) )
    print('the request took {0} seconds to complete'.format(time.time() - startOfRequest))

print('preliminary preparation for further inquiries took {0} seconds'.format(time.time() - startTime))
while True:
    s = input('Search : ')
    startOfRequest = time.time()
    result = search(s)
    printResult(result)