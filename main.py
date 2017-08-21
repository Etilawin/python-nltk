https://www.youtube.com/watch?v=FLZvOKSCkxY&list=PLQVvvaa0QuDf2JswnfiGkliBInZnIC4HL&index=1

from nltk.tokenize import sent_tokenize, word_tokenize

# Tokenizing - Word & Sentences tokenizers
# lecivon and corpras
# corpra - body of text. ex: medical journals, presidential speeches, English language
# lexicon - words and their means
#  Defintion depends on the context

example_text = "Hello there, how are you doing today ? The weather is great and Python is awesome. The sky is blue!"

##print(sent_tokenize(example_text))
##
##print(word_tokenize(example_text))

for i in word_tokenize(example_text):
    print(i)
