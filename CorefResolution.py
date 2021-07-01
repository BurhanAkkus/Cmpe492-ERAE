from pycorenlp import StanfordCoreNLP
import pprint
import copy
import os
def p(text):
    pprint.pprint(text)


def resolveSentence(sentence,depends):

    origSent=sentence
    #print(sentence)
    depends.sort()
    #ann['sentences'][mention['sentNum']-1]['tokens']
    newSent=[]
    #print(depends)
    last=0
    if(len(depends)==0):
        return sentence
    for depend in depends:
        newSent.extend(sentence[last:depend[0]])
        replaceString=depend[2]
        newSent.append(replaceString)
        last=depend[1]
    newSent.extend(sentence[last:len(sentence)])
    return(newSent)

tales=[]
with os.scandir('Stories/') as entries:
    for entry in entries:
        print(entry.name)
        if(entry.name!='Corefs'):
            tales.append(entry.name)
print(tales)

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
    #text = 'Barack was born in Hawaii. His wife Michelle was born in Milan. He says that she is very smart.'
    text=tale
    #print(f"Input text: {text}")

    def findRepresentative(chain):
        for mention in chain:
            #print((mention))
            if(mention['isRepresentativeMention']):
                return  mention['text']
    def p(text):
        pprint.pprint(text)
    # set up the client

    nlp_wrapper = StanfordCoreNLP('http://localhost:9000')
    doc = "Ronaldo has moved from Real Madrid to Juventus. While Messi still plays for Barcelona"
    doc=text
    #pprint.pprint(doc)
    annot_doc = nlp_wrapper.annotate(doc,
        properties={
            'annotators': 'dcoref,tokenize,ssplit,pos,lemma,ner,parse',
            'outputFormat': 'json',
            'timeout': 100000,
        })
    # submit the request to the server
    ann = annot_doc
    # submit the request to the server

    mychains = list()
    #p(ann['sentences'][0])
    chains = ann['corefs']
    #p(ann['sentences'][0])
    sentenceDeps=[[] for item in ann['sentences']]
    #p(chains)
    animates=[]
    for chain in chains:
        mychain = list()
        # Loop through every mention of this chain
    
        representativeText=findRepresentative(chains[chain])
        #p(chains[chain])
        if(chains[chain][0]['animacy']=='ANIMATE'):
            animates.append(representativeText)
            #p(chains[chain])
            for mention in chains[chain]:
                # Get the sentence in which this mention is located, and get the words which are part of this mention
                # (we can have more than one word, for example, a mention can be a pronoun like "he", but also a compound noun like "His wife Michelle")
                #p(mention)
                #print(mention['sentNum'],mention['startIndex'],mention['endIndex'])
                words_list = ann['sentences'][mention['sentNum']-1]['tokens'][mention['startIndex']-1:mention['endIndex']-1]
                sentenceDeps[mention['sentNum']-1].append([mention['startIndex']-1,mention['endIndex']-1,representativeText])
                #build a string out of the words of this mention
                #p(words_list)
                ment_word = ' '.join([x['word'] for x in words_list])
                mychain.append(ment_word)
            mychains.append(mychain)

    #for chain in mychains:
    # print(' <-> '.join(chain))
    #p(sentenceDeps)
    sentences=[]
    for sentence in ann['sentences']:
        sentences.append([x['word'] for x in sentence['tokens']])
    newSents=[]
    for idx,sentence in enumerate(sentences):
        newSents.append(resolveSentence(sentence,sentenceDeps[idx]))
    #p(newSents)
    f = open("./Corefs/"+taleName,'w',encoding="utf8")
    for sentence in newSents:
        for token in sentence:
            f.write(str(token))
            f.write(" ")
    f.close()
    f = open("./Animates/"+taleName,'w',encoding="utf8")
    for animate in animates:
            f.write(str(animate))
            f.write("\n")
    f.close()
