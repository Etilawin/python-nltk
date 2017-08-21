# https://www.youtube.com/watch?v=EymPQgCtcAE&index=6&list=PLQVvvaa0QuDf2JswnfiGkliBInZnIC4HL

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
        for i in tokenized:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)

            chunkGram = r""" Chunk: {<.*>+} # Chunking everything
                                    }<VB.?|IN|DT>+{"""   # Chinking

            chunkParser = nltk.RegexpParser(chunkGram)
            chunked = chunkParser.parse(tagged)

            chunked.draw()

    except Exception as e:
        print(str(e))

process_content()
