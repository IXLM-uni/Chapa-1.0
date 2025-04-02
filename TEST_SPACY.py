import spacy
text = ['Джон уехал в Москву 5 августа']
nlp = spacy.load('ru_core_news_md')

ner_labels = nlp.get_pipe('ner').labels
print(ner_labels)




