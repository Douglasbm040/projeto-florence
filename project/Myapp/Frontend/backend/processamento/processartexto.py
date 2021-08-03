import nltk

#base = [('eu sou admirada por muitos','alegria'),
#        ('me sinto completamente amado','alegria'),
#        ('amar e maravilhoso','alegria'),
#        ('estou me sentindo muito animado novamente','alegria'),
#        ('eu estou muito bem hoje','alegria'),
#        ('que belo dia para dirigir um carro novo','alegria'),
#        ('o dia está muito bonito','alegria'),
#        ('estou contente com o resultado do teste que fiz no dia de ontem','alegria'),
#        ('o amor e lindo','alegria'),
#        ('nossa amizade e amor vai durar para sempre', 'alegria'),
#        ('estou amedrontado', 'medo'),
#        ('ele esta me ameacando a dias', 'medo'),
#        ('isso me deixa apavorada', 'medo'),
#        ('este lugar e apavorante', 'medo'),
#        ('se perdermos outro jogo seremos eliminados e isso me deixa com pavor', 'medo'),
#        ('tome cuidado com o lobisomem', 'medo'),
#        ('se eles descobrirem estamos encrencados', 'medo'),
#        ('estou tremendo de medo', 'medo'),
#        ('eu tenho muito medo dele', 'medo'),
#        ('estou com medo do resultado dos meus testes', 'medo')]
base=[('prevenção de úceras por pressão','AGE'),
    ('Feridas com tecido de granulação','AGE'),
    #('feridas exudativas moderadas a altas','alginato de cálcio'),
    #('feridas com ou sem sangramentos','alginato de cálcio'),
    #('áreas doadoras de enxerto','alginato de cálcio'),
    #('feridas cavitária em geral','alginato de cálcio'),
    #('desbridamento de necrose de liquefaçào',"alginato de cálcio"),
    ('úlcera venosa e mista (arterial + venosa','bota de unna')
]
stopwords = nltk.corpus.stopwords.words('portuguese')

def removestopwords(texto):
    frases = []
    for (caracteristicas, material) in texto:
        semstop = [p for p in caracteristicas.split() if p not in stopwords]
        frases.append((semstop, material))
    return frases

def aplicastemmer(texto):
    stemmer = nltk.stem.RSLPStemmer()
    frasesstemming = []
    for (caracteristicas, material) in texto:
        comstemming = [str(stemmer.stem(p)) for p in caracteristicas.split() if p not in stopwords]
        frasesstemming.append((comstemming, material))
    return frasesstemming

frasescomstemming = aplicastemmer(base)

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

palavrasunicas = buscapalavrasunicas(frequencia)

def extratorpalavras(documento):
    doc = set(documento)
    caracteristicas = {}
    for palavra in palavrasunicas:
        caracteristicas['%s' % palavra] = (palavra in doc)
    return caracteristicas


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