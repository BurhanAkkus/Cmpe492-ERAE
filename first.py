import nltk
from pprint import pprint
from statistics import mean
from nltk.sentiment import SentimentIntensityAnalyzer
from random import shuffle
import matplotlib.pyplot as plt
import math
from tqdm import tqdm
import re
import string
from itertools import combinations
from collections import Counter
from flair.models import SequenceTagger
from flair.data import Sentence



def movingAverage(array,start,end):
    average=0
    for i in range(start,end+1):
        average=average+array[i]
    return average/(end-start+1)

sia = SentimentIntensityAnalyzer()


#tales=['HanselGretel','LittleRedRidingHood','ThreeLittlePigs','Cinderella','Golden Bird','HansInGoodLuck']
tales=['FundeVogel','Rapunzel','TheGooseGirl','Golden Bird','HansInGoodLuck','JorindaAndJorindel','TravelingMusicians','OldSultan','TheStraw','BriarRose','DogAndSparrow','TwelveDancingPrincesses','FishermanAndWife','TheWillowRen','FrogPrince','CatAndMouse']
taleSentiments=[]
for taleName in tales:
    f = open("./Stories/"+taleName+'.txt','r',encoding="utf8")

    tale= f.read()
    tale = tale.replace('\n', ' ')
    tale = tale.replace('\r', ' ')
    tale = tale.replace('\'', ' ')
    '''
    tagger = SequenceTagger.load('ner')
    x = []
    for line in tqdm(tale):
        sentence = Sentence(line)
        tagger.predict(sentence)
        for entity in sentence.to_dict(tag_type='ner')['entities']:
            if entity['type'] == 'PER':
             x.append(entity['text'])
    names = []
    for name in x:
        names.append(name.translate(str.maketrans('', '', string.punctuation)))
    
    result = [item for items, c in Counter(x).most_common() for item in [names] * c]
    #print(Counter(names).most_common())
    #print(x)
    '''
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
    plt.show()
   



'''''
tweets = [t.replace("://", "//") for t in nltk.corpus.twitter_samples.strings()]


def is_positive(review_id):
    text = nltk.corpus.movie_reviews.raw(review_id)
    scores = [
        sia.polarity_scores(sentence)["compound"]
        for sentence in nltk.sent_tokenize(text)
    ]
    return mean(scores) > 0

positive_review_ids = nltk.corpus.movie_reviews.fileids(categories=["pos"])
negative_review_ids = nltk.corpus.movie_reviews.fileids(categories=["neg"])
all_review_ids = positive_review_ids + negative_review_ids
shuffle(all_review_ids)
correct = 0
for review_id in all_review_ids:
    if is_positive(review_id):
       if review_id in positive_review_ids:
         correct += 1
    else:
        if review_id in negative_review_ids:
            correct += 1

print(F"{correct / len(all_review_ids):.2%} correct")


unwanted = nltk.corpus.stopwords.words("english")
unwanted.extend([w.lower() for w in nltk.corpus.names.words()])

def skip_unwanted(pos_tuple):
    word, tag = pos_tuple
    if not word.isalpha() or word in unwanted:
        return False
    if tag.startswith("NN"):
        return False
    return True

positive_words = [word for word, tag in filter(
    skip_unwanted,
    nltk.pos_tag(nltk.corpus.movie_reviews.words(categories=["pos"]))
)]
negative_words = [word for word, tag in filter(
    skip_unwanted,
    nltk.pos_tag(nltk.corpus.movie_reviews.words(categories=["neg"]))
)]


positive_fd = nltk.FreqDist(positive_words)
negative_fd = nltk.FreqDist(negative_words)

common_set = set(positive_fd).intersection(negative_fd)

for word in common_set:
    del positive_fd[word]
    del negative_fd[word]

top_100_positive = {word for word, count in positive_fd.most_common(100)}
top_100_negative = {word for word, count in negative_fd.most_common(100)}


unwanted = nltk.corpus.stopwords.words("english")
unwanted.extend([w.lower() for w in nltk.corpus.names.words()])

positive_bigram_finder = nltk.collocations.BigramCollocationFinder.from_words([
    w for w in nltk.corpus.movie_reviews.words(categories=["pos"])
    if w.isalpha() and w not in unwanted
])
negative_bigram_finder = nltk.collocations.BigramCollocationFinder.from_words([
    w for w in nltk.corpus.movie_reviews.words(categories=["neg"])
    if w.isalpha() and w not in unwanted
])
'''''