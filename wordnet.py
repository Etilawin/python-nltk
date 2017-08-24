# https://www.youtube.com/watch?v=T68P5-8tM-Y&index=10&list=PLQVvvaa0QuDf2JswnfiGkliBInZnIC4HL

# Wordnet is the largest capability corpora
# Look up synonyms and definitions

from nltk.corpus import wordnet

syns = wordnet.synsets("program")

# Synset
print(syns[0].name())

# Just the word itself
print(syns[0].lemmas()[0].name())

# definitions
print(syns[0].definition())

# Examples
print(syns[0].examples())

synonyms = []
antonyms = []

for syn in wordnet.synsets("good"):
    for l in syn.lemmas():
        synonyms.append(l.name())
        if l.antonyms():
            antonyms.append(l.antonyms()[0].name())

print(set(synonyms))
print(set(antonyms))

# Similarity

w1 = wordnet.synset("ship.n.01")
w2 = wordnet.synset("cactus.n.01")

print(w1.wup_similarity(w2)) # Pourcentages
