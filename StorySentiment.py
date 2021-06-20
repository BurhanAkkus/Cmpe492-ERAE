import nltk
from pprint import pprint
from statistics import mean
from nltk.sentiment import SentimentIntensityAnalyzer
from random import shuffle
import matplotlib.pyplot as plt
import math
import re
import string
from itertools import combinations
from collections import Counter
import os




def movingAverage(array,start,end):
    average=0
    for i in range(start,end+1):
        average=average+array[i]
    return average/(end-start+1)

sia = SentimentIntensityAnalyzer()
tales=[]
with os.scandir('Stories/') as entries:
    for entry in entries:
        print(entry.name)
        tales.append(entry.name)

taleSentiments=[]
for taleName in tales:
    f = open("./Stories/"+taleName,'r',encoding="utf8")

    tale= f.read()
    tale = tale.replace('\n', ' ')
    tale = tale.replace('\r', ' ')
    tale = tale.replace('\'', ' ')

    taleSentences = nltk.tokenize.sent_tokenize(tale)
    #pprint(tale)
    #pprint(taleSentences)

    sentiments=[]
    sentences=[]

    for sentence in taleSentences:
        if abs(sia.polarity_scores(sentence)['compound'])>0.1:
            sentiments.append(sia.polarity_scores(sentence)['compound'])
            sentences.append(sentence)
            #print(sentence,len(sentiments))
            #pprint(sia.polarity_scores(sentence)['compound'])

    movingAverages=[]
    start=0
    moveRate=len(sentiments)//10
    end=moveRate
    
    for i in range(len(sentiments)):
        #print(start,end)
        movingAverages.append(movingAverage(sentiments,start,end))
        end+=1
        if end==len(sentiments) or start+moveRate<=end:
            start+=1
        if(end==len(sentiments)):
            end=end-1
        if(start==end):
            break
        
    #print(movingAverages)
    print('Sentence Rate', len(sentiments)/len(taleSentences))
    #taleSentiments.append(movingAverages)
    plt.plot(movingAverages)
    #plt.plot(sentiments)
    plt.ylabel('Sentiment Averages'+ taleName)
    #plt.show()
    plt.savefig('./SentimentFigures/'+taleName[:-4]+'.png')
   