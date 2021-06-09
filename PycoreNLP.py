from pycorenlp import StanfordCoreNLP
import pprint
from collections import Counter
import os
tales=[]
with os.scandir('Corefs/') as entries:
    for entry in entries:
        print(entry.name)
        tales.append(entry.name)
def CountFrequency(my_list):
  
    # Creating an empty dictionary 
    freq = {}
    for item in my_list:
        if (item in freq):
            freq[item] += 1
        else:
            freq[item] = 1
  
    for key, value in freq.items():
        print ("% s : % d"%(key, value))

def p(tale):
    pprint.pprint(tale)


#tales=['FundeVogel','Rapunzel','TheGooseGirl','Golden Bird','HansInGoodLuck','JorindaAndJorindel','TravelingMusicians','OldSultan','TheStraw','BriarRose','DogAndSparrow','TwelveDancingPrincesses','FishermanAndWife','TheWillowRen','FrogPrince','CatAndMouse']
taleSentiments=[]
for taleName in tales:
    f = open("./Corefs/"+taleName,'r',encoding="utf8")
    p(taleName)
    tale= f.read()
    tale = tale.replace('\n', ' ')
    tale = tale.replace('\r', ' ')
    tale = tale.replace('\'', ' ')
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
    stopWords2= {'him','me','we','you','it','he','she','they','his','her','i','a','and','about','an','are','as','at','be','by','com','for','from','how','in','is','it','not','of','on','or','that','the','this','to','was','what','when','where','who','will','with','the','www','your','is','am','some','you','your','I','A','And','About','An','Are','As','At','Be','By','Com','For','From','How','In','Is','It','Not','Of','On','Or','That','The','This','To','Was','What','When','Where','Who','Will','With','The','Www','Your','Is','Am','Some','You','Your','Was'}
    nsubjs =[s for s in nsubjs if s.lower() not in stopWords2]

    result = [item for items, c in Counter(nsubjs).most_common()
                                        for item in [items] * c]
    CountFrequency(result)
    #pprint.pprint(result)