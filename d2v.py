from bsent import SentimentDoc2Vec

train_sources = {'dataset/train-neg.txt':'TRAIN_NEG', 'dataset/train-pos.txt':'TRAIN_POS'}
test_sources = {'dataset/test-neg.txt':'TEST_NEG', 'dataset/test-pos.txt':'TEST_POS'}
vocab_sources = {'dataset/train-neg.txt':'TRAIN_NEG', 'dataset/train-pos.txt':'TRAIN_POS', 'dataset/train-unsup.txt':'TRAIN_UNS'}

d2v = SentimentDoc2Vec(train_sources, test_sources, vocab_sources, 100)
d2v.make_model('./imdb.d2v')
d2v.make_dataset()
d2v.evaluate()
