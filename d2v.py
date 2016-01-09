from bsent import SentimentDoc2Vec

train_sources = {'train-neg.txt':'TRAIN_NEG', 'train-pos.txt':'TRAIN_POS'}
test_sources = {'test-neg.txt':'TEST_NEG', 'test-pos.txt':'TEST_POS'}
vocab_sources = {'train-neg.txt':'TRAIN_NEG', 'train-pos.txt':'TRAIN_POS', 'train-unsup.txt':'TRAIN_UNS'}

d2v = SentimentDoc2Vec(train_sources, test_sources, vocab_sources, 100)
d2v.make_model('./imdb.d2v')
d2v.make_dataset()
d2v.evaluate()
