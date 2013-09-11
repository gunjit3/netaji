import nltk
import util



feat_tokens = util.read_subj_obj_data();

classifier = nltk.NaiveBayesClassifier.train(feat_tokens[0])
sen = "it"
sentence = util.filter_stop_words(util.tokenize(sen))
featureset = util.get_feature(sentence, feat_tokens[1])
print classifier.classify(featureset)
print feat_tokens[0][-1]
#print classifier.show_most_informative_features(100)