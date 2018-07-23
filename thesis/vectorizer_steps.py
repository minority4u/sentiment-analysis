from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

import thesis.Data as d
import thesis.Vectorizer as vec

def Countvectorizer_steps():
    data = d.Data_loader().get_data()

    first_sentence = data['x_train'][0]

    raw_document = ['i loved the movie',
                'i hated the movie',
                'a great movie. good movie',
                'poor acting',
                'great acting. a good movie']

    raw_sentence = ['this is a test sentence']
    print('raw sentences:')
    print(raw_document)

    count_vec = CountVectorizer(binary=False)
    tokenized_sentences = count_vec.fit_transform(raw_document)
    print('tokenized sentences:')
    print(tokenized_sentences)
    print(count_vec.vocabulary_)


    # for sent in raw_document:
    #     print('raw sentence')
    #     print(sent)
    #     print(count_vec.transform([sent]))

    transformer = TfidfTransformer(sublinear_tf=True, use_idf=True)

    transformed_sentence = transformer.fit_transform(tokenized_sentences)
    print(transformed_sentence.data)
    for elem in transformed_sentence.data:
        print (elem)





if __name__ == "__main__":
    Countvectorizer_steps()