import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from scipy import sparse as sp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


vectorizer = CountVectorizer()
vectorizer
file = open('trainnew.txt', 'r').readlines()
X = vectorizer.fit_transform(file)
Y = X.toarray()

type(Y)
abc = TruncatedSVD(100)
Xabc = abc.fit_transform(Y)
#print(Xabc.explained_variance_ratio_)
plt.plot(Xabc)
plt.savefig('random9.png')
#pca= PCA(n_components=14)
#pca.fit(Y)
#pca.transform(Y)
#x = []
#y = []

#for ind_1, sublist in enumerate(Xabc):
#    for ind_2, ele in enumerate(sublist):
#        if ele == 1:
#            x.append(ind_1)
#            y.append(ind_2)

#sns.set()
#sns.pairplot(Y, size=2.5);
#plt.scatter(x,y)
#plt.savefig('nav.png')
