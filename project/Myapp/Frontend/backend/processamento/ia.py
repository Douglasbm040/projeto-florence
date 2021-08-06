import nltk
import numpy as np
import pandas as pd
from sklearn import *
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import LabelBinarizer
import re
from sklearn.neural_network import MLPClassifier

base=[('LPP','AGE'),
      ('UPP','AGE'),
      ('granulação','AGE'),
      ('exsudato moderado alto','Alginato de Cálcio'),
      ('enxerto','Alginato de Cálcio'),
      ('cavitárias','Alginato de Cálcio'),
      ('desbridamento','Alginato de Cálcio'),
      ('esfacelo','Alginato de Cálcio'),
      ('sangramentos','Alginato de Cálcio'),
      ('vaculares','Bota de Unna'),
      ('edema linfático','Bota de Unna'),
      
      ('odor','Carvão Ativado'),
    
      
      ('desvitalizado','Colagenase'),
      
      ('assaduras','Creme barreira'),
      ('dermatites','Creme barreira'),
      
      #('UPP','Filme Transparente'),
      #('LPP','Filme Transparente'),
      #('íntegra','Filme Transparente'),
      #('escoriações','Filme Transparente'),
      
      ('UPP','Hidrocolóide'),
      ('LPP','Hidrocolóide'),
      ('exsudato','Hidrocolóide'),
      ('cavitárias','Hidrocolóide'),
      
      ('exsudato','Hidrofibra'),
      
      ('vasculares','Hidrofibra'),
      ('diabéticas','Hidrofibra'),
      ('LPP ','Hidrofibra'),
      ('UPP','Hidrofibra'),
      ('queimaduras 2','Hidrofibra'),
      
      ('exsudato pouco','Hidrogel'),
      ('desvitalizados','Hidrogel'),
      ('enxerto','Hidrogel'),
      ('queimaduras 1  2','Hidrogel'),
      ('desbridamento','Hidrogel'),
      ('esfacelo','Hidrogel'),
      ('escara','Hidrogel'),
      
      ('granulação','papaina 2%'),
      ('esfacelo','papaina 6%'),
      ('escara','papaina 8%'),
      ('exsudativas','papaina'),
     
      ('colonizadas','PHMB'),
      ('queimaduras 1','PHMB'),
      ('queimaduras 2','PHMB'),
      ('UPP','PHMB'),
      ('LPP','PHMB'),
      ('vasculares','PHMB'),
      ('enxerto','PHMB'),
      ('infectada','PHMB'),

      ('vasculares','Polytube'),
      ('diabéticas','Polytube'),
      ('queimaduras 1','Polytube'),
      ('queimaduras 2','Polytube'),
      ('exsudativas','Polytube'),
      ('cavitárias ','Polytube'),

      ('infectada','Sulfadiazina de Prata 1%'),
      ('queimaduras 1','Sulfadiazina de Prata 1%'),
      ('queimaduras 2','Sulfadiazina de Prata 1%'),
      ('escaras','Sulfadiazina de Prata 1%'),
      ('vasculares','Sulfadiazina de Prata 1%'),
      
      

      ('enxerto','Petrolatum'),
      ('lacerações','Petrolatum'),
      ('abrasões','Petrolatum'),
      ('vasculares','VAC'),
      ('diabéticas','VAC'),
      ('traumáticas deiscências','VAC'),
      ('traumáticas deiscências LPP UPP diabéticas vasculares','VAC'),
      ('UPP','VAC'),
      ('LPP','VAC'),
]

stopwords = nltk.corpus.stopwords.words('portuguese')
contraindicacao=[
    ('esfacelo','AGE'),
    ('escara','AGE'),
    ('infectada','AGE'),
    ('seca','Alginato de Cálcio'),
    ('exudato','Alginato de Cálcio'),
    ('queimadura','Alginato de Cálcio'),
    ('ossos','Alginato de Cálcio'),
    ('tendões','Alginato de Cálcio'),
    ('estágio III','Alginato de Cálcio'),
    ('vasculares','Bota de Unna'),
    ('diabéticas','Bota de Unna'),
    ('queimaduras','Carvão Ativado'),
    ('hemorrágicas','Carvão Ativado'),
    ('escara','Carvão Ativado'),
    ('exsudato','Carvão Ativado'),
    ('granulação','Colagenase'),
    ('sudorese','Filme Transparente'),
    ('exsudato','Filme Transparente'),
    ('infectadas','Filme Transparente'),
    ('exsudato','Hidrocolóide'),
    ('infectadas','Hidrocolóide'),
    ('cavitárias','Hidrocolóide'),
    ('Região sacral','Hidrocolóide'),
    ('seca','Hidrofibra'),
    ('exsudato','Hidrogel'),
    ('pele íntegra','Hidrogel'),
    ('queimaduras 3','Hidrogel'),
    ('cartilagem hialina','PHMB'),
    ('queimaduras  3 4','PHMB'),
    ('cavidades que não tenha visualização da profundidade','PHMB'),
    ('estágio 3 III','PHMB'),
    ('gestantes no final da gestação','Sulfadiazina de Prata 1%'),
    ('crianças prematuras','Sulfadiazina de Prata 1%'),
    ('recém-natos nos 2 meses de vida','Sulfadiazina de Prata 1%'),
    ('infectadas','Petrolatum'),
    ('intensa grande exsudato','Petrolatum'),
    ('lesões tumorais fistulas entérica','VAC'),
    ('escara esfacelo tecido necrótico','VAC'),
   
]

curativo=[
    ('Trocar no máximo a cada 24 h ou sempre que o curativo secundário estiver saturado','AGE'),
    ('Feridas infectadas: no máximo a cada 24 h. Feridas limpas com sangramento: a cada 48h ou quando saturado; -Em outras situações a frequência das trocas deverá ser estabelecida de acordo com a avaliação do profissional que Havendo aumento do intervalo de trocas, devido à diminuição do exudato devese suspender o uso dessa cobertura para evitar o ressecamento do leito da ferida. acompanha o cuidado. -Considerar saturação do curativo secundário e aderência da cobertura no leito da ferida','Alginato de Cálcio'),
    ('-Troca a cada 7 dias. -Em caso de desconforto, vazamento de exsudato, sinais clínicos de infecção, dormência e latejamento dos dedos ou em caso de quaisquer outras irritações locais deve-se retirar a bandagem imediatamente','Bota de Unna'),
    ('A saturação do tecido de carvão ativado acontece, em média, em 3 a 4 dias, podendo ficar no leito até 7 dias. -Estabelecer necessidade de troca do curativo secundário conforme avaliação do profissional que acompanha o cuidado.','Carvão Ativado'),
    ('A cada 24 horas.','Colagenase '),
    ('-Resiste de 3 a 4 procedimento s de higiene. -Não é absorvido pela fralda ou lençóis.','Creme barreira'),
    ('Trocar no máximo a cada 7 dias e/ou quando necessário.','Filme Transparente'),
    ('Trocar no máximo a cada 7 dias, sempre que houver saturação da cobertura ou o curativo descolar','Hidrocolóide'),
    ('-Feridas limpas: até 7 dias; -Feridas infectadas: no máximo 3 dias; -Com prata: remover somente por vazamento, sangramento excessivo, dor ou em no máximo 7 dias.','Hidrofibra'),
    ('-Troca em até 48 horas. -Feridas infectadas: no máximo a cada 24 horas.','Hidrogel'),
    ('A frequência das trocas deverá ser estabelecida de acordo com a avaliação do profissional que acompanha o cuidado','Hidropolímero'),
    ('24 horas, antes se o curativo secundário estiver saturado.','Papaína'),
    ('-Mantém sua atividade em ambiente úmido por até 72 horas.','PHMB'),
    ('24 horas, antes se o curativo secundário estiver saturado.','papaina 2%'),
    ('24 horas, antes se o curativo secundário estiver saturado.','papaina 6%'),
    ('24 horas, antes se o curativo secundário estiver saturado.','papaina 8%'),
    ('Troca quando ocorrer 80%da saturação do produto. - Tempo máximo de permanência é de 7 dias.','Polytube'),
    ('-Feridas secas ou pouco exsudativas: troca em até 24 horas. -Feridas de muito exsudato: troca até 12h.','Sulfadiazina de Prata 1%'),
    ('-A frequência das trocas deverá ser estabelecida de acordo com a avaliação do profissional que acompanha o cuidado. -A saturação do curativo secundário e a possível aderência da cobertura no leito da ferida devem ser levados em consideração. -Pode permanecer até 7 dias em feridas limpas.','Petrolatum'),
    ('- Os curativos por terapia TPN devem ser trocados a cada 48 ou 72 horas (a frequência deve ser ajustada conforme o estado clínico do paciente);','VAC'),
    
    
]


def removestopwords(texto):
    frases = []
    for (descritor, cobertura) in texto:
        semstop = [p for p in descritor.split() if p not in stopwords]
        frases.append((semstop, cobertura))
    return frases 

def aplicastemmer(texto):
    stemmer = nltk.stem.RSLPStemmer()
    frasesstemming = []
    for (descritor, cobertura) in texto:
        comstemming = [str(stemmer.stem(p)) for p in descritor.split() if p not in stopwords]
        frasesstemming.append((comstemming, cobertura))
    return frasesstemming

frasescomstemming = aplicastemmer(base)

z = aplicastemmer(contraindicacao)


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
len(df['material'].unique())

for itens in range(0,len(df['ferida'])):
    s=df['ferida'][itens]
    out = re.sub(r'[^\w\s]','',s)
    df['ferida'][itens]=out

mlb = MultiLabelBinarizer()
mlb_result = mlb.fit_transform([str(df.loc[i,'ferida']).split(' ') for i in range(len(df))])
df_final = pd.concat([df['material'],pd.DataFrame(mlb_result,columns=list(mlb.classes_))],axis=1)

labelbinarizer = LabelBinarizer()
make_encoded_results = labelbinarizer.fit_transform(df_final['material'])
df_make = pd.DataFrame(make_encoded_results, columns=labelbinarizer.classes_)
df_make=df_make.join(df_final)


x=df_make['material'].unique().tolist()
y=df_make['material'].tolist()
y={'material':y}
y = pd.DataFrame.from_dict(y)
for i in range(0,len(x)):
 y[y['material']==x[i]]=i

y_train=df_make.loc[:, 'AGE':'papaina 8%']
x_train = df_make.drop(df_make.loc[:, 'AGE':'material'], 1)


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
delete=[]
teste = input('descreva : ')
testestemming = []
stemmer = nltk.stem.RSLPStemmer()
for (palavrastreinamento) in teste.split():
    comstem = [p for p in palavrastreinamento.split()]
    testestemming.append(str(stemmer.stem(comstem[0])))
search=testestemming
ci=[]
for j in testestemming:
    search=j
    for i in range(0,len(z)):
       if search == z[i][0][0]:
        ci.append(i)  

produtos=[]
for j in ci:
    produto=contraindicacao[j][1]
    produtos.append(produto)
    for i in range(0,len(base)):
      if produto == base[i][1]:
       delete.append(i)

fr=[]
for j in produtos:
    for i in range(0,len(frasescomstemming)):
       if j !=frasescomstemming[i][1]:
        fr.append(frasescomstemming[i])

basecompleta = nltk.classify.apply_features(extratorpalavras, fr)

classificador = nltk.NaiveBayesClassifier.train(basecompleta)




novo = extratorpalavras(testestemming)
naives=classificador.classify(novo)

print('naives receitou',naives)
#print('rede neural receitou',rede_neural )


for i in  range(0,len(curativo)):
    #if rede_neural == curativo[i][1] :
     #   
      #  print('A rede neural recomenda realizar curativo com : ',rede_neural)
       # print('OBS .:',curativo[i][0])
    if naives == curativo[i][1] :
        print('Para a descrição : ',teste)
        print('pelo naives realizar curatico com',naives,"obs:",curativo[i][0])
print('Descartados pela rede neural o uso de ',produtos)     
