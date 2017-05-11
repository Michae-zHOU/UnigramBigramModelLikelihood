import numpy as np
import math
import sys
import matplotlib.pyplot as plt

def qa(wordlist,wordcount,sum):
    print("\n---------Qa-------------")
    probability = []
    for i in range(0,len(wordlist)):
        probability.append(wordcount[i]/sum)
        if(wordlist[i][0] == 'M'):
            print(wordlist[i]+" "+str(wordcount[i]/sum))
    return probability

def qb(wordlist,wordcount,uniwordlist,uniprob):
    print("\n---------Qb-------------")
    one = []
    probability = []
    for i in range(0,len(wordlist)):
        index = uniwordlist.index(wordlist[i][1])
        probability.append(wordcount[i]/uniprob[index])
        if(wordlist[i][0] == 'ONE'):
            one.append([wordlist[i][1],wordcount[i]/uniprob[index]])
    one.sort(key = lambda o: o[1], reverse = True)
    for i in range(0,10):
        print(str(one[i][0])+" "+str(one[i][1]))
    return probability

def qc(uniwordlist,uniprob,biwordlist,biprob):
    print("\n---------Qc-------------")
    sentence = "The market fell by one hundred points last week".upper()
    ss = []
    for s in sentence.split(' '):
        ss.append(s)
    llu = 1
    indexbi = biwordlist.index(['<s>','THE'])
    llb = biprob[indexbi]
    for s in ss:
        index = uniwordlist.index(s)
        llu *= uniprob[index]
    for i in range(1,len(ss)):
        indexbi = biwordlist.index([ss[i-1],ss[i]])
        llb *= biprob[indexbi]
    llu = np.log(llu)
    llb = np.log(llb)
    print("llu: "+str(llu))
    print("llb: "+str(llb))
    if llu > llb:
        print('Unigram has highest log-likelihood.')
    else:
        print('Bigram has highest log-likelihood.')

def qd(uniwordlist,uniprob,biwordlist,biprob):
    print("\n---------Qd-------------")
    sentence = "The fourteen officials sold fire insurance".upper()
    ss = []
    for s in sentence.split(' '):
        ss.append(s)
    llu = 1
    indexbi = biwordlist.index(['<s>','THE'])
    llb = biprob[indexbi]
    for s in ss:
        index = uniwordlist.index(s)
        llu *= uniprob[index]
    for i in range(1,len(ss)):
        if [ss[i-1],ss[i]] not in biwordlist:
            print(str([ss[i-1],ss[i]])+" is not in the list.")
            llb *= 0
            biwordlist.append([ss[i-1],ss[i]])
            biprob.append(0)
        else:
            indexbi = biwordlist.index([ss[i-1],ss[i]])
            llb *= biprob[indexbi]
    print("The log-likelihood from the bigram model is negative infinity.")

def qe(uniwordlist,uniprob,biwordlist,biprob):
    print("\n---------Qe-------------")
    sentence = "The fourteen officials sold fire insurance".upper()
    ss = []
    for s in sentence.split(' '):
        ss.append(s)
    lamda = np.arange(0.,1.,0.01)
    lm = []
    maxLam = 0.
    maxres = -sys.maxsize
    for l in lamda:
        res = calculateLm(ss,uniwordlist,uniprob,biwordlist,biprob,l)
        lm.append(res)
        if res > maxres:
            maxLam = l
            maxres = res

    print("Max Lamda: "+str(maxLam))
    plt.figure("Likelihood Maximum Graph")
    plt.xlabel("Lambda")
    plt.ylabel("Lm")
    plt.plot(lamda,lm)
    plt.show()


def calculateLm(ss,uniwordlist,uniprob,biwordlist,biprob,lamda):
    indexu = uniwordlist.index('THE')
    indexb = biwordlist.index(['<s>','THE'])
    Lm = (1-lamda)*(uniprob[indexu])+lamda*(biprob[indexb])
    for i in range(1,len(ss)):
        Pm = (1-lamda)*(uniprob[uniwordlist.index(ss[i])]) + lamda*(biprob[biwordlist.index([ss[i-1],ss[i]])])
        Lm *= Pm
    Lm = np.log(Lm)
    return Lm


def main():
    f1 = open('vocab.txt', 'r')
    f2 = open('unigram.txt', 'r')
    f3 = open('bigram.txt', 'r')
    uniwordlist = []
    biwordlist = []
    uniwordcount = []
    biwordcount = []
    uniprob = []
    biprob = []
    unigramSum = 0
    bigramSum = 0
    for vocab in f1:
        if vocab[len(vocab)-1] == '\n':
            uniwordlist.append(vocab[0:len(vocab)-1])
        else:
            uniwordlist.append(vocab)
    for count in f2:
        c = int(count)
        uniwordcount.append(c)
        unigramSum += c
    for line in f3:
        inp = line.split('\t')
        input = [int(inp[0])-1,int(inp[1])-1,int(inp[2])]
        biwordlist.append([uniwordlist[input[0]],uniwordlist[input[1]]])
        biwordcount.append(input[2])
        bigramSum += input[2]

    uniprob = qa(uniwordlist,uniwordcount,unigramSum)
    biprob = qb(biwordlist,biwordcount,uniwordlist,uniwordcount)
    qc(uniwordlist,uniprob,biwordlist,biprob)
    qd(uniwordlist,uniprob,biwordlist,biprob)
    qe(uniwordlist,uniprob,biwordlist,biprob)

if __name__ == "__main__":
    main()
