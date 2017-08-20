# https://www.youtube.com/watch?v=w36-U-ccajM&index=2&list=PLQVvvaa0QuDf2JswnfiGkliBInZnIC4HL

# Stop words -> when you see it you jump it

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

example_sentence = "This is an example showing of stop words filtration."
stop_words = set(stopwords.words("english"))

words = word_tokenize(example_sentence)
#
# filtered_sentence = []
#
# for w in words:
#     if(w not in stop_words):
#         filtered_sentence.append(w)
#
# print(filtered_sentence)

filtered_sentence = [w for w in words if not w in stop_words] # Same as b4

print(filtered_sentence) # Great fun
