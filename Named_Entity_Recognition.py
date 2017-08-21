# https://www.youtube.com/watch?v=LFXsG7fueyk&list=PLQVvvaa0QuDf2JswnfiGkliBInZnIC4HL&index=7

# chink something from a chunk

import nltk
from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer # AI to tokenize

train_text = state_union.raw("2005-GWBush.txt")
sample_text = state_union.raw("2006-GWBush.txt")

custom_sent_tokenize = PunktSentenceTokenizer(train_text) # Learns how to tokenize

tokenized = custom_sent_tokenize.tokenize(sample_text) # Tokenizes w/ what it learned

def process_content():
    try:
        for i in tokenized[5:]:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)

            namedEnt = nltk.ne_chunk(tagged, binary=True)

            namedEnt.draw()

    except Exception as e:
        print(str(e))

process_content()
