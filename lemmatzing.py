# https://www.youtube.com/watch?v=uoHVztKY6S4&index=8&list=PLQVvvaa0QuDf2JswnfiGkliBInZnIC4HL
# Looks like stemming -> but working

from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

print(lemmatizer.lemmatize("better", pos="a"))
print(lemmatizer.lemmatize("best", pos="a"))
print(lemmatizer.lemmatize("run"))
print(lemmatizer.lemmatize("ran", pos="v"))
