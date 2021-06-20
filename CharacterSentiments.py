from pycorenlp import StanfordCoreNLP
import pprint
from collections import Counter
from nltk.sentiment import SentimentIntensityAnalyzer
import os
import sys
tales=[]
with os.scandir('Corefs/') as entries:
    for entry in entries:
        print(entry.name)
        tales.append(entry.name)

def movingAverage(array,start,end):
    average=0
    for i in range(start,end+1):
        average=average+array[i]
    return average/(end-start+1)
    
def tokenToText(tokens):
    text=''
    for token in tokens:
        text=text+" "+str(token)
    return text

sia = SentimentIntensityAnalyzer()
def CountFrequency(my_list,top,taleName):
  
    # Creating an empty dictionary 
    freq = {}
    print(top)
    for item in my_list:
        if (item.lower() in freq):
            freq[item.lower()] += 1
        else:
            freq[item.lower()] = 1
    i=0
    characters=[]
    f=open("./Characters/"+taleName,'w',encoding="utf8")
    for key, value in freq.items():
        #print ("% s : % d"%(key, value))
        f.write("% s : % d \n"%(key, value))
        characters.append(key)
        i=i+1
        if(i>=top):
            break
    f.close()
    return characters

def p(tale):
    pprint.pprint(tale)

def sentenceInvolvesCharacter(sentence,character):
    text=tokenToText(sentence)
    #p(text)
    #p(character)
    #p(text.find(character) != -1)
    if text.lower().find(character) != -1:
        return True
    return False

def GetCharacterSentiment(character,sentences):
    sentiments=[]
    sum=0
    for sentence in sentences:
        #print(sentence)
             sentiment=sia.polarity_scores(sentence)['compound']
             sentiments.append(sentiment)
             sum=sum+sentiment
    if(len(sentiments)==0):
        return 0
    return sum/(len(sentiments))

#tales=['FundeVogel','Rapunzel','TheGooseGirl','Golden Bird','HansInGoodLuck','JorindaAndJorindel','TravelingMusicians','OldSultan','TheStraw','BriarRose','DogAndSparrow','TwelveDancingPrincesses','FishermanAndWife','TheWillowRen','FrogPrince','CatAndMouse']
taleSentiments=[]
for taleName in tales:
    #f = open("./Corefs/"+taleName,'r',encoding="utf8")
    p(taleName)
    if (sys.argv[1]==1):
        f=open("./Stories/"+taleName,'r',encoding="utf8")
    else:
        f=open("./Corefs/"+taleName,'r',encoding="utf8")
    tale= f.read()
    tale = tale.replace('\n', ' ')
    tale = tale.replace('\r', ' ')
    #pprint.pprint(tale)
    nlp_wrapper = StanfordCoreNLP('http://localhost:9000')
    #doc = "Ronaldo has moved from Real Madrid to Juventus. While Messi still plays for Barcelona"
    doc=tale
    #pprint.pprint(doc)
    annot_doc = nlp_wrapper.annotate(doc,
        properties={
            'annotators': 'ner, pos,depparse',
            'outputFormat': 'json',
            'timeout': 100000,
        })

    nsubjs=[]
    #pprint.pprint(annot_doc)
    for sentence in annot_doc['sentences']:
        for element in sentence['basicDependencies']:
            if(element['dep']=='nsubj'):
                nsubjs.append(element['dependentGloss'])
    stopWords2= {'him','me','we','you','it','he','she','all','one','my','they','his','her','i','a','and','about','an','are','as','at','be','by','com','for','from','how','in','is','it','not','of','on','or','that','the','this','to','was','what','when','where','who','will','with','the','www','your','is','am','some','you','your','I','A','And','About','An','Are','As','At','Be','By','Com','For','From','How','In','Is','It','Not','Of','On','Or','That','The','This','To','Was','What','When','Where','Who','Will','With','The','Www','Your','Is','Am','Some','You','Your','Was'}
    nsubjs =[s for s in nsubjs if s.lower() not in stopWords2]

    result = [item for items, c in Counter(nsubjs).most_common()
                                        for item in [items] * c]
    characters=CountFrequency(result,len(annot_doc['sentences'])/10,taleName)
    #pprint.pprint(result)
    characterSentiments=[]
    for character in characters:
        relevantSentences=[]
        for sentence in annot_doc['sentences']:
            #p(sentence)
            sentenceTokens=[x['word'] for x in sentence['tokens']]
            #p(sentenceTokens)
            if(sentenceInvolvesCharacter(sentenceTokens,character)):
                relevantSentences.append(tokenToText(sentenceTokens))
            #p(relevantSentences)
            #p(character)
        characterSentiments.append([character,GetCharacterSentiment(character,relevantSentences)])
    p(characterSentiments)
    f=open("./Sentiments/"+taleName,'w',encoding="utf8")
    for item in characterSentiments:
        #print ("% s : % d"%(key, value))
        f.write((item[0]+" : "+ str(item[1]))+"\n")
    f.close()