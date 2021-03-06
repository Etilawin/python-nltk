# https://www.youtube.com/watch?v=yGKTphqxR9Q&index=3&list=PLQVvvaa0QuDf2JswnfiGkliBInZnIC4HL

# Stemming = recherche du radical
# Stemming => Look for the stem of a word

# Why ? => Simplify

# I was taking a ride in the car
# I was riding in the car
# The same meaning but different expression of ride

from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

ps = PorterStemmer()

example_words = {"python", "pythoner", "pythoning", "pythoned", "pythonly"}

# for w in example_words:
#     print(ps.stem(w))

new_text = "It is very import to be pythonly while you are pythoning with python. All pythoners have pythoned poorly at least once."

words = word_tokenize(new_text)

for w in words:
    print(ps.stem(w))
