import operator
import math
import string
import matplotlib.pyplot as py
import numpy as np
import sys
def SUM_BI_COUNT(parent):
    sum = 0
    for (word1, word2) in BI_COUNT.keys():
        if word1 == parent:
            sum = sum + BI_COUNT[(parent, word2)]

    return sum


def ML_PROB_U(word):
    word = word.upper()
    if word not in UNI_COUNT:
        word = '<UNK>'
    
    return UNI_COUNT[word]/SUM_UNI_COUNT

def ML_PROB_B(word1, word2):
    word1 = word1.upper()
    word2 = word2.upper()
    if word1 not in VOCAB:
        word1 = '<UNK>'
    if word2 not in VOCAB:
        word2 = '<UNK>'
    if (word1, word2) not in BI_COUNT.keys():
        #print(word1, ' ', word2, ' combination does not exist')
        return 0
    
    return BI_COUNT[(word1, word2)]/SUM_BI_COUNT(word1)


def LL_U(sentence):
    exclude = set(string.punctuation)
    sentence = ''.join(ch for ch in sentence if ch not in exclude)
    sentence = sentence.split(" ")
    print(sentence)
    Pu = 1
    for word in sentence:
        Pu = Pu * ML_PROB_U(word)
    
    if Pu == 0:
        print("Undefined bigram log likelihood")
        return
    
    print("Log likelihood for unigram model is ", math.log(Pu), '\n')


def LL_B(sentence):
    exclude = set(string.punctuation)
    sentence = ''.join(ch for ch in sentence if ch not in exclude)
    sentence = sentence.split(" ")
    print(sentence)
    Pu = 1
    sentence.insert(0,'<S>')
    for i in range(1, len(sentence)):
        Pu = Pu * ML_PROB_B(sentence[i-1], sentence[i])
    
    if Pu == 0:
        print("Undefined bigram log likelihood", '\n')
        return
    
    print("Log likelihood for bigram model is ", math.log(Pu), '\n')


def LL_M(sentence):
    exclude = set(string.punctuation)
    sentence = ''.join(ch for ch in sentence if ch not in exclude)
    sentence = sentence.split(" ")
    print(sentence)
    sentence.insert(0,'<S>')
    
    lambda1 = []
    probArray=[]
    uu = (-sys.maxsize)
    ll = 0
    for l in range(0,101):
        Pu = 1
        l = l*0.01
        lambda1.append(l)
        for i in range(1, len(sentence)):
            Pu = Pu * (l* ML_PROB_U(sentence[i]) + (1-l)*ML_PROB_B(sentence[i-1], sentence[i]))
        probArray.append(np.log(Pu))
        if np.log(Pu) > uu:
            ll = l
            uu = np.log(Pu)
    print(ll)
    print("------")
    py.plot(lambda1,probArray,linewidth=2.0)
    py.show()




# open the file hw3_vocab.txt and create a list of the tokens
file = open("hw4_vocab.txt")
VOCAB = file.readlines()
VOCAB = list(map(lambda s: s.strip('\n'), VOCAB))

# open the file hw3_unigram.txt and create a list of the tokens
file = open("hw4_unigram.txt")
UNI_COUNT = file.readlines()
UNI_COUNT = list(map(lambda s: int(s.strip('\n')), UNI_COUNT))
UNI_COUNT = dict(zip(VOCAB, UNI_COUNT))

#  PART a : maximum likelihood estimate of the unigram distribution Pu(w) over words w
SUM_UNI_COUNT = sum(UNI_COUNT.values())
# Print out a table of all the tokens (i.e., words) that start with the letter “M”, along with their numerical unigram
# probabilities (not counts).
for token in UNI_COUNT:
    if token[0] == 'M':
        print(token,' ', ML_PROB_U(token))

print('\n')

# PART b :
BI_COUNT = {}
# open the file hw3_bigram.txt and create a list of the tokens
file = open("hw4_bigram.txt")
lines = file.readlines()
for line in lines:
    line = line.strip('\n')
    entry = line.split('\t')
    BI_COUNT.update({(VOCAB[int(entry[0])-1], VOCAB[int(entry[1])-1]) : int(entry[2])})


# Print out a table of the ten most likely words to follow the word “THE”, along with their numerical bigram probabilities.
THE = {}
for (w1,w2) in BI_COUNT:
    if w1 == 'ONE':
        THE.update({w2:ML_PROB_B(w1, w2)})

SORTED_THE = list(sorted(THE.items(), key=operator.itemgetter(1), reverse=True))
print("10 most frequent tokens after THE are: ")
for tuple in SORTED_THE[0:10]:
    print(tuple)

print('\n')

# PART C
LL_U("The market fell by one hundred points last week")
LL_B("The market fell by one hundred points last week")

# plot the mixture model
LL_M("The fourteen officials sold fire insurance")
