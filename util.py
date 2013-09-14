import nltk

def tokenize(sentences) :
    "this will take a list of sentences and return a list of list of token"
    "corresponding to each sentence"
    "Input ['a sentence','second sentence']"
    "output[['a','sentence],['second','sentence']]"
    return [nltk.word_tokenize(s) for s in sentences]

def remove_stop_words(tokens):
    "this will take a list of tokens and return the list with stop words removed"
    stops = set(nltk.corpus.stopwords.words('english'))
    return [t for t in tokens if not t in stops and t.isalnum()]

def filter_stop_words(sentence_tokens) :
    "this will take a list of list with tokens corresponding to each sentence and will"
    "return a list of list with stop words removed"
    return [remove_stop_words(s) for s in sentence_tokens]

def sentence_tokens_to_tokens(sentence_tokens):
    "this will take list of list of tokens and return tokens"
    return [t for s in sentence_tokens for t in s]

def read_file(filename):
    "reads a file and return a list of line"
    return [line.strip().lower() for line in open(filename)]

def get_feature(sentence, corpus_tokens):
    "return a feature set that is passed to classifier based on"
    "sentence, all tokens in the corpus"
    features = {}
    for word in corpus_tokens:
        features['contains(%s)' % word] = (word in sentence)
    return features

def read_subj_obj_data():
    "reads subjective and objective data from respective file and maps it to the "
    "Corresponding class. Prepares a feature vector as used by classfier and returns it"
    "this needsmore work"
    
    subj_file = "/home/gunjit/workspace/Twitter/dataset/quote.tok.gt9.5000"
    obj_file = "/home/gunjit/workspace/Twitter/dataset/plot.tok.gt9.5000"
    subj_data = read_file(subj_file)
    obj_data = read_file(obj_file)
    subj_tokens = filter_stop_words(tokenize(subj_data[:100]))
    obj_tokens = filter_stop_words(tokenize(obj_data[:100]))
    all_tokens = sentence_tokens_to_tokens(subj_tokens)
    all_tokens.append(sentence_tokens_to_tokens(obj_tokens))
    features = []
    for sentence in subj_tokens :
        sent_feat = get_feature(sentence, all_tokens)
        features.append((sent_feat,"Subjective"))
    for sentence in obj_tokens :
        sent_feat = get_feature(sentence, all_tokens)
        features.append((sent_feat,"Objective"))
    return [features, all_tokens]

    