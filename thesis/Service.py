
import logging
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords

import thesis.IO_Organizer as io
import thesis.my_logger

class API_Service(object):
    def __init__(self):
        self.classifier = self.__load_classifier__()
        self.vectorizer = self.__load_vectorizer__()
        pass


    def __load_classifier__(self):
        return io.load_classifier('LinearSVC')

    def __load_vectorizer__(self):
        return io.load_vectorizer('TfidfVectorizer')

    def __review_to_words__(self, raw_review):
        # logging.info('cleaning data')
        # Function to convert a raw review to a string of words
        # The input is a single string (a raw movie review), and
        # the output is a single string (a preprocessed movie review)
        #
        # 1. Remove HTML
        review_text = BeautifulSoup(raw_review, 'html.parser').get_text()
        #
        # 2. Remove non-letters
        letters_only = re.sub("[^a-zA-Z]", " ", review_text)
        #
        # 3. Convert to lower case, split into individual words
        words = letters_only.lower().split()
        #
        # 4. In Python, searching a set is much faster than searching
        #   a list, so convert the stop words to a set
        stops = set(stopwords.words("english"))
        #
        # 5. Remove stop words
        # meaningful_words = [w for w in words if not w in stops]
        meaningful_words = words
        #
        # 6. Join the words back into one string separated by space,
        # and return the result.
        # logging.info('cleaning data - done')
        return (" ".join(meaningful_words))


    def single_text_inference(self, document='this is a test sentence, the movie was very great.'):
        # manuel function to clean a review
        # This could be done by regex tokenizer or word tokenizer from nltk

        raw_document = document
        logging.info('raw')
        logging.info(raw_document)

        cleaned_document = self.__review_to_words__(raw_document)
        logging.info('cleaned')
        logging.info(cleaned_document)

        vectorized_document = self.vectorizer.transform([cleaned_document])
        logging.info('vectorized')
        logging.info(vectorized_document)

        prediction = self.classifier.predict(vectorized_document)
        logging.info('prediction')
        logging.info(prediction)





def test_inference():

    inference_sentence = "this movie doesn't suck, i loved it."

    sa_service = API_Service()
    sa_service.single_text_inference(inference_sentence)


if __name__ == "__main__":
    test_inference()