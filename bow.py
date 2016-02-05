from bsent import SentimentBoW

train_sources = {'dataset/train-neg.txt':'TRAIN_NEG', 'dataset/train-pos.txt':'TRAIN_POS'}
test_sources = {'dataset/test-neg.txt':'TEST_NEG', 'dataset/test-pos.txt':'TEST_POS'}
vocab_sources = {'dataset/train-neg.txt':'TRAIN_NEG', 'dataset/train-pos.txt':'TRAIN_POS', 'dataset/train-unsup.txt':'TRAIN_UNS'}

bow = SentimentBoW(train_sources, test_sources, vocab_sources)
bow.make_dataset()
bow.evaluate()

