#!/usr/bin/env python
# coding: utf-8

# In[74]:


import nltk
import numpy as np
import pandas as pd
from sklearn import *
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import LabelBinarizer
import re


# In[125]:


base=[('upp','AGE'),
   ('granulação','AGE'),
   ('exudativas','alginato de cálcio'),
   ('sangramentos','alginato de cálcio'),
   ('enxerto','alginato de cálcio'),
   ('cavitária','alginato de cálcio'),
   ('desbridamento de necrose de liquefação',"alginato de cálcio"),
   ('venosa e arterial UV UA','bota de unna'),
      ('infectadas com ou sem odor profundas com exsudação','Carvão Ativado'),
      ('tecido desvitalizado','Colagenase'),
      ('assaduras dermatites pele seca ou irritada','creme barreira'),
]

stopwords = nltk.corpus.stopwords.words('portuguese')


# In[126]:


def removestopwords(texto):
    frases = []
    for (palavras, emocao) in texto:
        semstop = [p for p in palavras.split() if p not in stopwords]
        frases.append((semstop, emocao))
    return frases 

print(removestopwords(base))

# completamente
# completo
# comp


# In[127]:


#retirar a raiz da palavra
def aplicastemmer(texto):
    stemmer = nltk.stem.RSLPStemmer()
    frasesstemming = []
    for (palavras, emocao) in texto:
        comstemming = [str(stemmer.stem(p)) for p in palavras.split() if p not in stopwords]
        frasesstemming.append((comstemming, emocao))
    return frasesstemming

frasescomstemming = aplicastemmer(base)
frasescomstemming


# In[128]:


lista=frasescomstemming
dataset={}
material=[]
ferida=[]
for i in range(0,len(lista)):
    material.append(lista[i][1])
    ferida.append(str(lista[i][0]))

dataset={'material':material,'ferida':ferida}
df = pd.DataFrame.from_dict(dataset, orient='index')
df = df.transpose()
type(df['ferida'][0])


# In[129]:




for itens in range(0,len(df['ferida'])):
    s=df['ferida'][itens]
    out = re.sub(r'[^\w\s]','',s)
    df['ferida'][itens]=out
df['ferida']

    


# In[130]:


a=np.eye(6)
lst=[]
for x in a:
    lst.append(x)
type(lst)


# In[131]:


mlb = MultiLabelBinarizer()
mlb_result = mlb.fit_transform([str(df.loc[i,'ferida']).split(' ') for i in range(len(df))])
df_final = pd.concat([df['material'],pd.DataFrame(mlb_result,columns=list(mlb.classes_))],axis=1)
mlb.classes_


# In[132]:


# Enoding make column using LabelBinarizer

labelbinarizer = LabelBinarizer()
make_encoded_results = labelbinarizer.fit_transform(df_final['material'])
df_make = pd.DataFrame(make_encoded_results, columns=labelbinarizer.classes_)
df_make=df_make.join(df_final)
df_make


# In[133]:


listaunicadematerial=df_make['material'].unique().tolist()
listaunicadematerial


# In[134]:


x=df_make['material'].unique().tolist()
y=df_make['material'].tolist()
y={'material':y}
y = pd.DataFrame.from_dict(y)
for i in range(0,len(x)):
 y[y['material']==x[i]]=i
y


# In[135]:


df_make
y_train=df_make.loc[:, 'AGE':'creme barreira']
x_train = df_make.drop(df_make.loc[:, 'AGE':'material'], 1)

x_train


# In[170]:


def buscapalavras(frases):
    todaspalavras = []
    for (caracteristicas, material) in frases:
        todaspalavras.extend(caracteristicas)
    return todaspalavras

caracteristicas = buscapalavras(frasescomstemming)

def buscafrequencia(caracteristicas):
    caracteristicas = nltk.FreqDist(caracteristicas)
    return caracteristicas

frequencia = buscafrequencia(caracteristicas)

def buscapalavrasunicas(frequencia):
    freq = frequencia.keys()
    return freq

palavrasunicas = mlb.classes_

def extratorpalavras(documento):
    doc = set(documento)
    caracteristicas = {}
    for palavra in palavrasunicas:
        caracteristicas['%s' % palavra] = (palavra in doc)
    return caracteristicas


# In[171]:



basecompleta = nltk.classify.apply_features(extratorpalavras, frasescomstemming)

classificador = nltk.NaiveBayesClassifier.train(basecompleta)

teste = 'úlcera venoso em membro inferior'
testestemming = []
stemmer = nltk.stem.RSLPStemmer()
for (palavrastreinamento) in teste.split():
    comstem = [p for p in palavrastreinamento.split()]
    testestemming.append(str(stemmer.stem(comstem[0])))

novo = extratorpalavras(testestemming)
print(classificador.classify(novo))


# In[194]:


testestemming
basecompleta = nltk.classify.apply_features(extratorpalavras, frasescomstemming)

classificador = nltk.NaiveBayesClassifier.train(basecompleta)

teste = 'ua'
testestemming = []
stemmer = nltk.stem.RSLPStemmer()
for (palavrastreinamento) in teste.split():
    comstem = [p for p in palavrastreinamento.split()]
    testestemming.append(str(stemmer.stem(comstem[0])))

novo = extratorpalavras(testestemming)
print(classificador.classify(novo))
master=novo.values()


# In[195]:


master=list(master)
type(master)


# In[196]:


k=[]
for i in master:
    if i == True:
     k.append(1)
    else:
     k.append(0)


# In[197]:


len(k)


# In[198]:


type(master[0])


# In[199]:


k =np.array(k)
k = np.reshape(k, (-1, 24))


# In[200]:


len(novo)


# In[201]:


k


# In[202]:


mlb.classes_==novo


# In[203]:


from sklearn import datasets 
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score
#from yellowbrick.classifier import ConfusionMatrix 

iris = datasets. load_iris()
y_train=y.astype('int')



modelo = MLPClassifier (verbose = False, hidden_layer_sizes=(24,6), max_iter = 10000) 
modelo.fit(x_train, y_train)



previsoes = modelo.predict(k) 
accuracy_score(k, previsoes)



#confusão=ConfusionMatrix(modelo) 
#confusao.fit(x_treinamento, y_treinamento)



#confusao.score(x_teste, y_teste) 
#confusao.poof()


# In[204]:





# In[148]:


y_train


# In[ ]:




