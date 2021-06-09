from pycorenlp import StanfordCoreNLP
import pprint
text = 'Barack was born in Hawaii. His wife Michelle was born in Milan. He says that she is very smart.'
print(f"Input text: {text}")


def p(text):
    pprint.pprint(text)
# set up the client

nlp_wrapper = StanfordCoreNLP('http://localhost:9000')
doc = "Ronaldo has moved from Real Madrid to Juventus. While Messi still plays for Barcelona"
doc=text
#pprint.pprint(doc)
annot_doc = nlp_wrapper.annotate(doc,
    properties={
        'annotators': 'coref',
        'outputFormat': 'json',
        'timeout': 100000,
    })
# submit the request to the server
ann = annot_doc
# submit the request to the server

mychains = list()
p(ann)
chains = ann['corefs']
for chain in chains:
    mychain = list()
    # Loop through every mention of this chain
    for mention in chain:
        # Get the sentence in which this mention is located, and get the words which are part of this mention
        # (we can have more than one word, for example, a mention can be a pronoun like "he", but also a compound noun like "His wife Michelle")
        words_list = ann.sentence[mention.sentenceIndex].token[mention.beginIndex:mention.endIndex]
        #build a string out of the words of this mention
        ment_word = ' '.join([x.word for x in words_list])
        mychain.append(ment_word)
    mychains.append(mychain)

for chain in mychains:
    print(' <-> '.join(chain))