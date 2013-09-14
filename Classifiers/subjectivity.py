import nltk
import sys
import util

SUBJECTIVITY_DATA = "/home/gunjit/workspace/Twitter/quote.tok.gt9.5000";

class subjectivity :
    def __init__(self):
        "initialize the class"
        self.trained = 0
        self.classifier = None
        pass
    def train(self):
        "train the classifier based on a dataset"
        if self.trained == 1:
            return
        feature_tokens = util.read_subj_obj_data();
        self.classifier = nltk.NaiveBayesClassifier.train(feature_tokens[0])
        self.all_tokens = feature_tokens[1]
        self.trained = 1
    def predict(self, sentence):
        "predict a sentence based on trained classifier"
        if self.trained == 0:
            print >> sys.stderr, 'please train the classifier first'
            return None
        sent_feat = util.get_feature(sentence, self.all_tokens)
        return self.classifier.classify(sent_feat)
        
        
        
