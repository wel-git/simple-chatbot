'''
IMPORTS AND CONFIGURATIONS
'''
import random
import json
import pickle
import numpy as np

import os



absolute_path = os.path.abspath(__file__)

directory_path = os.path.dirname(absolute_path)


intents = json.loads(open(os.path.join(directory_path , 'data/intents.json')).read())



import nltk


from nltk import SnowballStemmer
snowball = SnowballStemmer(language='english')

from keras.models import load_model


words = pickle.load(open(os.path.join(directory_path,"./pickle/words.pkl"), 'rb'))

tags = pickle.load(open(os.path.join(directory_path,"./pickle/tags.pkl"), 'rb'))

model = load_model (os.path.join(directory_path, "model"))



def clean_up_sentence(sentence): 
    '''
    ARGS
        sentence
            user input

    RETURN
        user input that has been split, tokenised, and stemmed (SnowballStemmer) for future uses
    '''
    tokenised_list = nltk.tokenize.word_tokenize(sentence)

    stemmed_tokenised_list = [snowball.stem(tokenised_word.lower()) for tokenised_word in tokenised_list]

    return stemmed_tokenised_list


def bag_of_words (sentence): 
    '''
    ARGS 
        sentence
            user input
    
    RETURN 
        binary numpy array that is a cleaned up user input where len(bag) == len(words)
    '''
    sentence_words = clean_up_sentence(sentence)

    bag = []
    for word in words: 
        bag.append(1) if word in sentence_words else bag.append(0)

    # print(f"this is bag: {bag}")

    return np.array (bag)


def predict_class (sentence): 
    '''
    ARGS
        sentence
            user input
    
    RETURN 
        result
            type: dictionary
            example: [
                {"intent_index": 3, "probability": 0.9487}, 
                {"intent_index": 2, "probability": 0.5487}, 
                {"intent_index": 4, "probability": 0.3487}, 
                {"intent_index": 0, "probability": 0.2587},]
            explanation: 
                ["intent_index"] corresponds to the index of tags
                ["probability"] shows the probability of the user input to be the corresponding tag

            rejects any probability below ERROR_THRESHOLD = 0.25
    '''
    bow = bag_of_words(sentence)
    
    all_res = model.predict(np.array([bow]))[0]
    


    ERROR_THRESHOLD = 0.25

    raw_unsorted_dict = {}
    class_index = 0
    for res in all_res: 

        if res > ERROR_THRESHOLD:
            raw_unsorted_dict [class_index] = res
        class_index += 1
    
    

    # print(raw_unsorted_dict)
    sorted_dict=dict(sorted(raw_unsorted_dict.items(),key= lambda x:x[1], reverse=True))
    
    
    result = [{"intent_index": index, "probability": sorted_dict[index]} for index in sorted_dict ]

    return result
    

def get_response (predict_class_dict, intents_json): 
    '''
    ARGS
        predict_class_dict
            type: dictionary 
            see: <function predict_class>
        intents_json
            type: dictionary
            read intents.json file

    RETURN
        random string from intents[tag_name] ["responses"]
        output for user to see
    '''
    tag_index = predict_class_dict[0]['intent_index']
    list_of_intents = intents_json["intents"]
    
    responses = list_of_intents [tag_index]["responses"]
    random_number = random.randrange(1, len(responses))
    result = responses [random_number]
    return result


print("CHATBOT IS RUNNING")
while True: 
    message = input()
    ints = predict_class(message)
    res = get_response(ints, intents)
    print(res)