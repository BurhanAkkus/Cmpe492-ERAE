import nltk
#nltk.download('vader_lexicon')

print("Original Text:")
print("Hansel and Gretel are trapped in the witch's house. They have to find a way out before she catches them!")
print("Text after Coreference Resolution")
print("Hansel and Gretel are trapped in the witch's house. Hansel and Gretel have to find a way out before the witch catches Hansel and Gretel!")

from nltk.sentiment import SentimentIntensityAnalyzer

sia = SentimentIntensityAnalyzer()
print(sia.polarity_scores("Poor Granny did not have time to say a word, before the wolf gobbled her up!​"))