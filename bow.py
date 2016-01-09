from bsent import SentimentBoW

train_sources = {'train-neg.txt':'TRAIN_NEG', 'train-pos.txt':'TRAIN_POS'}
test_sources = {'test-neg.txt':'TEST_NEG', 'test-pos.txt':'TEST_POS'}
vocab_sources = {'train-neg.txt':'TRAIN_NEG', 'train-pos.txt':'TRAIN_POS', 'train-unsup.txt':'TRAIN_UNS'}

bow = SentimentBoW(train_sources, test_sources, vocab_sources)
bow.make_dataset()
bow.evaluate()

