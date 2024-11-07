import joblib
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.datasets import fetch_20newsgroups

newsgroups_data = fetch_20newsgroups(subset='train')
texts = newsgroups_data.data

documents = [TaggedDocument(words=text.split(), tags=[str(i)]) for i, text in enumerate(texts)]

model = Doc2Vec(vector_size=50, min_count=2, epochs=40)
model.build_vocab(documents)
model.train(documents, total_examples=model.corpus_count, epochs=model.epochs)

model.save("doc2vec_model.model")
print("Doc2Vec модель успішно збережена як 'doc2vec_model.model'")
