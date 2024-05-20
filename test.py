import numpy as np
import pandas as pd

BOS = "<s>"
EOS = "</s>"
SEP = "<sep>"
PAD = "<pad>"
UNK = "<unk>"

def make_aggregate(vocab_size, dataset_size, min_length=4, max_length=16, seed=0):
    vocab = np.array([str(i) for i in range(vocab_size - 3)])
    sents, tags = [], []
    np.random.seed(seed)
    
    for _ in range(dataset_size):
        l = np.random.randint(min_length, max_length)
        sent = np.random.choice(vocab, size=l, replace=True).astype(int).tolist()
        aggregate_value = sum(sent) / len(sent)
        aggregated_sent = [round(aggregate_value, 1)] * l
        
        sents.append([BOS] + list(map(str, sent)) + [EOS])
        tags.append([PAD] + list(map(str, aggregated_sent)) + [PAD])
    
    return pd.DataFrame({"sent": sents, "tags": tags})

def make_sort(vocab_size, dataset_size, min_length=4, max_length=16, seed=0):
    vocab = np.array([str(i) for i in range(vocab_size - 3)])
    sents, tags = [], []
    np.random.seed(seed)
    for _ in range(dataset_size):
        l = np.random.randint(min_length, max_length - 1)
        sent = np.random.choice(vocab, size=l, replace=True).tolist()
        sents.append([BOS] + sent + [EOS])
        tags.append([PAD] + sorted(sent) + [PAD])
    return pd.DataFrame({"sent": sents, "tags": tags})

def make_join(vocab_size, dataset_size, min_length=4, max_length=16, seed=0):
    # Specified vocabulary of salaries
    vocab = [62000, 69000, 80000, 60000, 75000]

    np.random.seed(seed)
    sents, tags = [], []
    vocab_size = len(vocab)
    
    for _ in range(dataset_size):
        # Generate a sentence of salaries from the given vocabulary
        sent = np.random.choice(vocab, size=vocab_size).tolist()
        # Generate exactly vocab_size random bonus indices
        bonus_indices = np.random.randint(0, vocab_size, size=vocab_size).tolist()
        
        # Calculate new salaries based on bonus counts
        bonus_counts = np.bincount(bonus_indices, minlength=vocab_size)
        new_salaries = [salary + bonus_counts[idx] * 5000 for idx, salary in enumerate(sent)]
        
        # Convert all elements to strings
        sent_str = list(map(str, sent + bonus_indices))
        tags_str = list(map(str, new_salaries + bonus_indices))

        # Combine salaries and bonus indices for "sent" and "tags"
        sents.append([BOS] + sent_str + [EOS])
        tags.append([PAD] + tags_str + [PAD])
    
    return pd.DataFrame({"sent": sents, "tags": tags})

#make_aggregate(8, 10000)
#make_sort(8, 10000)
make_join(10, 10000)
