from pycorenlp import StanfordCoreNLP
import nltk
import pprint
import json
import copy

f = open("./Stories/HanselGretel.txt",'r',encoding="utf8")
tale= f.read()
tale = tale.replace('\n', ' ')
tale = tale.replace('\r', ' ')


text=tale
#text = 'Barack was born in Hawaii. His wife Michelle was born in Milan. He says that she is very smart.'
print(f"Input text: {text}")
'''
>>> l = [1, 2, 3, 4, 5]
>>> l[2:2] = ['a', 'b', 'c']
>>> l
[1, 2, 'a', 'b', 'c', 3, 4, 5]
'''
def p(text):
    pprint.pprint(text)

def findRepresentative(chain):
    for mention in chain:
        #print((mention))
        if(mention['isRepresentativeMention']):
            return  copy.deepcopy(mention)

def isPossessive(text):
    if(text.lower() in ['his','her']):
        return True
    return False

def replaceMention(tokenized,representatives):
    for representative in representatives:
        #p(representative['mentions'])
        #p(representative['text'])
        for mention in representative['mentions']:
            sentNum=mention[0]-1
            start,end= mention[1]-1,mention[2]-1
            #p(sentNum)
            #p(start)
            #p(end)
            #p(tokenized[sentNum][start:end])
            #p(len(tokenized[sentNum]))
            #p(tokenized[sentNum])
            #p(representative['text'])
            repWord=tokenize_text(representative['text'])[0]
            #p(repWord)
            tokenized[sentNum][start:end]=repWord
            #p(tokenized[sentNum])
            # for i in range(len(tokenized[sentNum])):
            #     p(i)
        
    p(tokenized)
        #tokenized[sentNum][start:end]=[representativeText]

def representativeDependencies(representatives):
    dependencies=[[]  for item in representatives]
    positions=[[x['sentNum'],x['startIndex'],x['endIndex']] for x in representatives]
    #print(positions)
    for index,reper in enumerate(representatives):
        #pprint.pprint(reper)
        for mention in reper['mentions']:
            for dependent in representatives:
                    if( mention[0]==dependent['sentNum'] and mention[1]>=dependent['startIndex'] and mention[2]<dependent['endIndex']):
                        dependent['dependent'].append([reper['chainID'],isPossessive(mention[3]),mention[1],mention[2],index])

def resolveRepresentativeDependencies(representatives):
    while(not allResolved(representatives)):
        #solve Recursively
        for representative in representatives:
            #pprint.pprint(representative)
            resolveRepresentativeDependency(representative,representatives)
            #print('ASDASD')

def resolveRepresentativeDependency(representative,representatives):
   
    representative['dependent'].sort(key = lambda x: x[2])
    if(representative['dependent']==[]):
        return
    
    lastIndex=-representative['dependent'][-1][3]+representative['endIndex']
    representedText=representatives[representative['dependent'][0][-1]]['text']
    p(representedText)
    newText=''
    tokens=tokenize_text(representative['text'])
    tokens=tokens[0]
    pprint.pprint(representative)
    p(tokens)
    for i in range(lastIndex):
        j=i+1
        if(len(representative['dependent'])==0):
            break
        representedStart=representative['dependent'][0][2]-representative['startIndex']
        representedEnd=-representative['dependent'][0][3]+representative['endIndex']
        p('START')
        print(representedStart)
        print(representedEnd)
        p('END')
        if(j<representedStart):
            #print(len(tokens))
            #print(i)
            newText=newText+' '+tokens[i]
        if(j==representedEnd):
            resolveRepresentativeDependency(representatives[representative['dependent'][0][4]],representatives)
            newText=newText+representatives[representative['dependent'][0][4]]['text']
            if(representative['dependent'][0][1]):
                newText=newText+'\'s'
            representative['dependent'].pop(0)
        print(j,newText)
    #p(range(lastIndex,len(tokens)))     
    for i in range(lastIndex,len(tokens)):
        #print(i)
        newText=newText+' '+tokens[i]
    representative['text']=newText
    print(newText)

def allResolved(representatives):
    for representative in representatives:
        #print(representative)
        if(representative['dependent']!=[]):
            return False
    return True

def tokenize_text(text):
    token_sen = nltk.sent_tokenize(text)
    word = []
    for i in range(len(token_sen)):
        word.append(nltk.word_tokenize(token_sen[i]))
    return word
# set up the client
nlp_wrapper = StanfordCoreNLP('http://localhost:9000')
doc = "Ronaldo has moved from Real Madrid to Juventus. While Messi still plays for Barcelona"
doc=text
#pprint.pprint(doc)
annot_doc = nlp_wrapper.annotate(doc,
    properties={
        'annotators': 'coref',
        'coref.algorithm' : 'statistical',
        'outputFormat': 'json',
        'timeout': 100000,
    })
# submit the request to the server
ann = annot_doc
#pprint.pprint(ann)
mychains = list()
chains = ann['corefs']
#pprint.pprint(chains)
#print(type(chains))
#for key in chains:
#   print(key, '->', chains[key])
tokenized=tokenize_text(text)

'''for sentence in tokenized:
    for word in sentence:
        #print(word)
        print(word,isPossessive(word))
'''
replaced=[]
representatives=[]
for key in chains:
    represent=findRepresentative(chains[key])
    represent['mentions']=[[x['sentNum'],x['startIndex'],x['endIndex'],x['text']] for x in chains[key]]
    represent['chainID']=key
    represent['dependent']=[]
    representatives.append(represent)
#pprint.pprint(representatives)
representativeDependencies(representatives)
#pprint.pprint(representatives)
resolveRepresentativeDependencies(representatives)

pprint.pprint(representatives)

    # reper=representatives[i]['text']
    # print(reper)
#replaceMention(tokenized,representatives)
    #replaced.append(True)
#print(tokenized)
#print(representatives)

#pprint.pprint(ann)

'''
for chain in chains:
    mychain = list()
    # Loop through every mention of this chain
    for mention in chain['mention']:
        # Get the sentence in which this mention is located, and get the words which are part of this mention
        # (we can have more than one word, for example, a mention can be a pronoun like "he", but also a compound noun like "His wife Michelle")
        words_list = ann.sentence[mention.sentenceIndex].token[mention.beginIndex:mention.endIndex]
        #build a string out of the words of this mention
        ment_word = ' '.join([x.word for x in words_list])
        mychain.append(ment_word)
    mychains.append(mychain)

for chain in mychains:
    print(' <-> '.join(chain))
    '''