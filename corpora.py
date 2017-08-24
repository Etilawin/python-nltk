# https://www.youtube.com/watch?v=TKAXDqoG2dc&list=PLQVvvaa0QuDf2JswnfiGkliBInZnIC4HL&index=9

# Corpora access & Edition
# D:\Python\Python36-32\nltk_data\corpora

from nltk.corpus import gutenberg
from nltk.tokenize import sent_tokenize

sample = gutenberg.raw("bible-kjv.txt")
tok = sent_tokenize(sample);

print(tok[5:15])
