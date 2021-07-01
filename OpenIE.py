from pycorenlp import StanfordCoreNLP
import pprint
import os
from nltk.sentiment import SentimentIntensityAnalyzer

sia = SentimentIntensityAnalyzer()

def p(text):
    pprint.pprint(text)




tales=[]
with os.scandir('Stories/') as entries:
    for entry in entries:
        print(entry.name)
        if(entry.name!='Corefs'):
            tales.append(entry.name)
nlp_wrapper = StanfordCoreNLP('http://localhost:9000')
doc = "Grandma,Ronaldo and Anna has moved from Real Madrid to Juventus. While Messi still plays for Barcelona. He loves Ronaldo and Natalie!!"
doc="Then the cat took a walk upon the roofs of the town , looked out for opportunities , and then stretched the cat in the sun , and licked the cat's lips whenever the cat thought of the pot of fat , and not until it was evening did the cat return home ."
doc="the cat went straight to the church , stole the pot of fat , began to lick at it , and licked the top of the fat off ."
punctuations=['.',',','\'','!','?',':',';']
for taleName in tales:
    f = open("./Stories/"+taleName,'r',encoding="utf8")

    tale= f.read()
    tale = tale.replace('\n', ' ')
    tale = tale.replace('\r', ' ')
    tale = tale.replace('‘', '\"')
    tale = tale.replace('’', '\"')
    tale = tale.replace('“', '\"')
    tale = tale.replace('”', '\"')
    for punctuation in punctuations:
        tale = tale.replace(punctuation,punctuation+ ' ')
    characterFile = open("./Characters/"+taleName, "r")
    characters=[]
    characters1 = characterFile.readlines()
    for line in characters1:
        line = line.replace('\n', '')
        characters.append(line)
    
    characterFile.close()
    relationsDict=dict()
    for character in characters:
        for character2 in characters:
            relationsDict[(character,character2)]=[]
    #p(characters)
    #pprint.pprint(doc)
    p(taleName)
    annot_doc = nlp_wrapper.annotate(tale,
        properties={
            'annotators': 'tokenize,ssplit,pos,lemma,depparse,natlog,openie',
            'outputFormat': 'json',
            "openie.triple.strict":"true",
            'timeout': 100000,
        })
    # submit the request to the server
    ann = annot_doc
    openie=[]
    sentenceDicts=[]
    #print(characters)
    for sentence in ann['sentences']:
        #p(sentence['basicDependencies'])
       
        tokens=[word['word'] for word in sentence['tokens']]
        sentenceString=''
        for token in tokens:
                   #p(token)
            sentenceString=sentenceString+token+' '
        #p(sentenceString.lower())
        
        sentenceCharacters=[character for character in characters if character in sentenceString.lower()]
        #p(sentenceCharacters)
        #p(sentenceString)
        for character1 in sentenceCharacters:
            for character2 in sentenceCharacters:
                relationsDict[(character1,character2)].append([sia.polarity_scores(sentenceString)['compound'],sentenceString])
        sentenceDicts.append(dict())
        '''
        for relation in sentence['openie']:
            
            object=relation['object'].lower().split()[-1]
            if relation['subject'].lower() in characters and object in characters:
            
                
            #p(relation)
                tokens=[word['word'] for word in sentence['tokens']]
                sentenceString=''
                for token in tokens:
                    #p(token)
                    sentenceString=sentenceString+token+' '
                #p(sentenceString)
                if ((relation['subject'],object)not in sentenceDicts[-1]):
                    sentenceDicts[-1][(relation['subject'],object)]=[relation['relation'],sia.polarity_scores(relation['subject']+" "+relation['relation']+" "+object),]
                    relationsDict[(relation['subject'].lower(),object)].append((sia.polarity_scores(sentenceString),sentenceString))

                    #p(sentenceString)
                    #p(relation)
                    #p(object)
                else:
                    sentenceDicts[-1][(relation['subject'],object)].append([relation['relation'],sia.polarity_scores(relation['subject']+" "+relation['relation']+" "+object)])

            #if (len(sentence['openie'])>0 and sentence['openie'][0]['subject']=='Hansel'):
            # 
            openie.append([[relation['subject'],relation['object'],relation['relation']] for relation in sentence['openie']])
            '''
        


    #p(openie)
    relations=[]
    #p(sentenceDicts)
    #p(relationsDict)

    relationScores=dict()
    for character in characters:
        for character2 in characters:
            sum=0
            flag=True
            num=0
            for relation in relationsDict[character,character2]:
                sum=sum+relation[0]
                flag=False
                num=num+1
            if(not flag):
                relationScores[(character,character2)]=sum/num
    sortedRelations = sorted(relationScores.items(), key=lambda x: x[1], reverse=True)

    p(sortedRelations)
    #for idx,relation in enumerate(openie):
    #    p(relation[0]+" "+relation[2]+" "+relation[1])
    
    f=open("./Relations/"+taleName,'w',encoding="utf8")
    # for relation in relationScores:
    #     f.write(str(relation))
    #     f.write("\n")
    f.write(str(sortedRelations))
    # for item in characterSentiments:
    #     print ("% s : % d"%(key, value))
    #         f.write((item[0]+" : "+ str(item[1]))+"\n")
    # f.close()
        
