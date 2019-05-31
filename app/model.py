import json
import os
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from modAL.models import ActiveLearner

from utils import *

#Load Data

data_dir = os.path.abspath('')

with open('/home/tnaraindas/work/CHARR/recipes.json','r') as f:
    recipes = json.load(f)
    
with open('/home/tnaraindas/work/CHARR/seed_recipes.json','r') as f:
    seed = json.load(f)
    
x_seed, y_seed = construct_df(seed, label=True)
x_pool  = construct_df(recipes)[0]

#Vectorize recipes
vec = TfidfVectorizer(stop_words='english',ngram_range=(1,1),tokenizer=tokenize)
vec.fit(x_seed.append(x_pool))

x_seed_vec = vec.transform(x_seed)
x_pool_vec = vec.transform(x_pool)

#Build learner
#pipe = Pipeline([('tfidf', TfidfVectorizer(stop_words='english',ngram_range=(1,1),tokenizer=tokenize)),
#                    ('model',LogisticRegression(C=1))])

learner = ActiveLearner(estimator=LogisticRegression(C=10,solver='lbfgs'), X_training=x_seed_vec,
                        y_training=np.array(y_seed).reshape(len(y_seed),-1))

for n in range(5):
    
    query_idx, query_inst = learner.query(x_pool_vec)
    
    recipe = x_pool[query_idx].index.values[0]
    print(recipes[recipe]['recipe'])
    print('0/1?')
    response = int(input())
    print('\n')
    learner.teach(x_pool_vec[query_idx].reshape(1, -1), np.array(response).reshape(1, -1))
    
#Return recommeded recipe

pred = learner.predict_proba(x_pool_vec)[:,1]
max_pred = np.argmax(pred)
rec = x_pool.index[max_pred]

print('Your recommended recipe is:')
print(recipes[rec]['recipe'])
