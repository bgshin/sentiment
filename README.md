# sentiment
Sentiment Analysis on IMDB dataset, (BoW, d2v)

## pip install

```bash
numpy==1.10.2
scipy==0.16.1
gensim==0.12.3
scikit-learn==0.17
```

## usage

* sentiment analysis using bag of words

```python
from bsent import SentimentBoW

train_sources = {'train-neg.txt':'TRAIN_NEG', 'train-pos.txt':'TRAIN_POS'}
test_sources = {'test-neg.txt':'TEST_NEG', 'test-pos.txt':'TEST_POS'}
vocab_sources = {'train-neg.txt':'TRAIN_NEG', 'train-pos.txt':'TRAIN_POS', 'train-unsup.txt':'TRAIN_UNS'}

bow = SentimentBoW(train_sources, test_sources, vocab_sources)
bow.make_dataset()
bow.evaluate()
```

*  sentiment analysis using dock2vec

```python
from bsent import SentimentDoc2Vec

train_sources = {'train-neg.txt':'TRAIN_NEG', 'train-pos.txt':'TRAIN_POS'}
test_sources = {'test-neg.txt':'TEST_NEG', 'test-pos.txt':'TEST_POS'}
vocab_sources = {'train-neg.txt':'TRAIN_NEG', 'train-pos.txt':'TRAIN_POS', 'train-unsup.txt':'TRAIN_UNS'}

d2v = SentimentDoc2Vec(train_sources, test_sources, vocab_sources, 100)
d2v.make_model('./imdb.d2v')
d2v.make_dataset()
d2v.evaluate()
```
