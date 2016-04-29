import os, pickle, re, sys
import numpy as np
import scipy.sparse



vocab_size = 3000
data_path = "../data/MaskedDataRaw.csv"
vocab_path = "../data/vocal_3000.pkl"
save_path = "../data/bagOfWods_3000.npz"

emoticons_str = r"""
    (?:
        [:=;] # Eyes
        [oO\-]? # Nose (optional)
        [D\)\]\(\]/\\OpP] # Mouthr
    )"""

regex_str = [
    emoticons_str,
    r'<[^>]+>', # HTML tags
    r'(?:@[\w_]+)', # @-mentions
    r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)", # hash-tags
    r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+',
    # URLs

    r'(?:(?:\d+,?)+(?:\.?\d+)?)', # numbers
    r"(?:[a-z][a-z'\-_]+[a-z])", # words with - and '
    r'(?:[\w_]+)', # other words
    r'(?:\S)' # anything else
]

tokens_re = re.compile(
        r'('+'|'.join(regex_str)+')', re.VERBOSE | re.IGNORECASE)
emoticon_re = re.compile(r'^'+emoticons_str+'$', re.VERBOSE | re.IGNORECASE)

_DIGIT_RE = re.compile(r"\d")

def tokenize(s):
    """This function tokenize a sentence into a list of words/token"""
    return tokens_re.findall(s)

def preprocess(s, lowercase=False):
    """This function tokenizes a sentence and keep emoticon"""
    tokens = tokenize(s)
    if lowercase:
        tokens = [token if emoticon_re.search(token) else token.lower() for \
                   token in tokens]
    return tokens


def create_vocabulary(data_path, vocab_path, vocab_size,
			normalize_digits, header):
    """Creating the vocabulary of words. It then keeps the most popular words,
    with size given by vocab_size
    It then save the vocabulary to disk"""
    print("Creating vocabulary at %s" % vocab_path)
    vocab = {}
    with open(data_path) as f:
        if header: f.readline()
        for i, line in enumerate(f.readlines()):
            if i % 10000 == 0:
                sys.stdout.write(str(i) + ".")
                sys.stdout.flush()
            word_list = preprocess(','.join(line.split(',')[3:]), True)
            for word in word_list:
                # word = word.lower()
                if word.isdigit(): word = re.sub(_DIGIT_RE, "0", word) if \
                        normalize_digits else word
                if word in vocab:
                    vocab[word] += 1
                else:
                    vocab[word] = 1
    vocab_list = sorted(vocab, key=vocab.get, reverse=True)
    vocab_list = vocab_list[:vocab_size]
    pickle.dump(vocab_list, open(vocab_path, 'wb'))
    return vocab

def initialize_vocabulary(data_path, vocab_path, vocab_size,
        normalize_digits, header):
    """Load the vocabulary saved to disk, and create a dictionary and a reversed
    dictionary"""
    if not os.path.isfile(vocab_path):
        create_vocabulary(data_path, vocab_path, vocab_size,
                normalize_digits, header)
    rev_vocab = pickle.load(open(vocab_path, 'rb'))
    vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
    return vocab, rev_vocab

def tweet_to_count_data(data_path, vocab_path, vocab_size,
                                            normalize_digits, header):
    """Create a count matrix, for number of row equal to number of movies, and
    number of columns equal to number of words in the vocabulary. Each entry at
    row i column j denote how many time word j'th appear in tweets of movie i'th
    """
    vocab, rev_vocab = initialize_vocabulary(data_path, vocab_path,
            vocab_size, normalize_digits, header)

    n = 0
    with open(data_path) as f:
	    for line in f.readlines(): n+=1
    if header:
        n -= 1
    p = len(vocab)
    X = scipy.sparse.lil_matrix((n, p), dtype = 'uint8')
    print("Creating the count matrix (%d, %d)" %(n, p))

    with open(data_path) as f:
        if header: f.readline()
        for i, line in enumerate(f.readlines()):
            if i % 1000 == 0:
                sys.stdout.write(str(i) + '.')
                sys.stdout.flush()
            word_list = preprocess(','.join(line.split(',')[3:]), True)
            for word in word_list:
                # word = word.lower()
                if word.isdigit(): word = re.sub(_DIGIT_RE, "0", word) if \
                        normalize_digits else word
                if word in vocab:
                    X[i, vocab[word]] += 1
    return X

def main():
    if not os.path.exists(save_path):
        X = tweet_to_count_data(data_path, vocab_path, 
                vocab_size, True, True)
        X = X.tocsc()
        with open(data_path) as f:
            f.readline()
            y = [int(line.split(',')[1]) for line in f.readlines()]
            y = np.array(y)
        np.savez(save_path, X = X, y = y)

if __name__ == "__main__":
    main()