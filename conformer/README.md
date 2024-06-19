# Conformer replication

The code for generating the weights used for replicating the CONFORMER is included in [this](./pregenerate_conformer_bias.ipynb) notebook.
The notebook includes the code for training the [LDA-model](./lda_model.zip), in addition to both the feed-forward networks using BoW frequency to estimate both [Word Weights](./BoW_Freq_To_Weights.pt) and [Topic Weights](./BoW_Freq_To_Topics.pt).

## Using the LDA model

To use the included LDA model, it needs to be loaded using the `Gensim` library:

```python
import gensim
lda = gensim.models.LdaModel.load('cnn_lda/lda', mmap='r')
dct = gensim.corpora.Dictionary.load('cnn_lda/filtered_extremes.dic')
```

All inference using the LDA model and feed-forward network is based on the included tokenizer and `Gensim`-dictionary's `doc2bow`.
Generating the `BoW` is done by:

```python
cnn_with_tokens = cnn.map(tokenize_dataset)

def add_gensim_bow(x):
    x['article_gensim_bow'] = dct.doc2bow(x['article_tokens'])
    x['highlights_gensim_bow'] = dct.doc2bow(x['highlights_tokens'])
    return x

cnn_with_bow = cnn_with_tokens.map(add_gensim_bow)

```

## Using the feed-forward models

The included feed-forward models must be loaded using the same configuration as was used during training.
These are:

### BoW_Freq_To_Weights

```python
model_config = {
    "input_size": 10_000,
    "hidden_size": 300,
    "output_size": 10_000,
    "hidden_activation_function": None,
    "output_activation_function": None,
    "loss": torch.nn.CrossEntropyLoss,
    "dropout": None,
    "optimizer": torch.optim.Adam,
}

model = LitTopicFeedForward(**model_config)
model.load_state_dict(torch.load('./BoW_Freq_To_Weights.pt'))
```

### BoW_Freq_To_Topics

```python
model_config = {
    "input_size": 10_000,
    "hidden_size": 300,
    "output_size": 250,
    "hidden_activation_function": None,
    "output_activation_function": None,
    "loss": torch.nn.CrossEntropyLoss,
    "dropout": None,
    "optimizer": torch.optim.Adam,
}

model = LitTopicFeedForward(**model_config)
model.load_state_dict(torch.load('./BoW_Freq_To_Topics.pt'))
```

To use both models, the input data needs to be processed like the notebook:

```python
import numpy as np
from scipy.sparse import coo_matrix
from tqdm import tqdm
from sklearn.preprocessing import normalize


def get_whole_sparse_matrix(bow_iterable, norm=False):
    indices = []
    values = []

    for row_index, row_pairs in enumerate(tqdm(bow_iterable)):
        for col_index, value in row_pairs:
            indices.append((int(row_index), int(col_index)))
            values.append(value)
    row_indices, col_indices = zip(*indices)

    shape = (max(row_indices) + 1, max(col_indices) + 1)
    coo_mat = coo_matrix((values, (row_indices, col_indices)), shape=shape, dtype=np.float32)
    if norm:
        return normalize(coo_mat, norm="l1")
    return coo_mat

input_data = {
    'train':get_whole_sparse_matrix(cnn_with_bow =['train']['article_gensim_bow'], norm=True).toarray(),
    'validation':get_whole_sparse_matrix(cnn_with_bow =['validation']['article_gensim_bow'], norm=True).toarray(),
    'test': get_whole_sparse_matrix(cnn_with_bow =['test']['article_gensim_bow'], norm=True).toarray()
}

model(input_data['test'])
```

In addition, to generate the resulting word weights from both the LDA model and the feed-forward net for topics, the result needs to be multiplied by the topic word distribution:

```python
topic_word_dist = lda.get_topics()

word_weights_lda = lda.get_document_topics(cnn_with_bow['test']['article_gensim_bow']) @ topic_word_dist
word_weights_feed_forward = model(input_data['test']) @ topic_word_dist
```