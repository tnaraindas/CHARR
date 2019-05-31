import pandas as pd
import re
import spacy

nlp = spacy.load('en_core_web_sm')

def construct_df(d, label=False):
    
    '''Create model-ready df from json'''
    
    x = []
    y = []
    for keys, values in d.items():

        temp = values['ingredients']+' '+values['instructions']
        x.append(temp)
        
        if label:
            y.append(values['label'])

    #x = pd.DataFrame(x,index = recipes.keys(), columns = ['text'])
    x = pd.Series(x,index = d.keys())
    if label:
        y = pd.Series(y,index = d.keys())
    else:
        y = None
    
    return x, y


def tokenize(X):
    
    '''Custom word tokenizer'''
    
    #cust_stop_words = []
    stop_patterns = ['^http','^www','^/d+']
    stop_patterns = '|'.join(stop_patterns)
    
    tokens = nlp(X)
    
    tokens = [t for t in tokens if (not t.is_punct) & (t.pos_ in ['NOUN','VERB','ADJ','PROPN'])]
    tokens = [t.lemma_.lower().strip() for t in tokens if len(t) > 2]
    tokens = [t for t in tokens if not re.search(stop_patterns,t)]

    return tokens

