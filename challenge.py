

# %%
import spacy
nlp = spacy.load("de_core_news_lg")
import json
import warnings
warnings.filterwarnings("ignore")


# %%
with open ("data.json",'r', encoding="utf-8") as f:  # added encoding method
    data = json.load(f)


# %%
# Turn data into format that could be accepted during training
from utils import transform_data
prepared_data = transform_data(data)

# %%
# check the distribution of all entitiy labels
# note that this is not count through the number of data, but through the number of occurence
# the result shows the occurence for each entity types are imbalanced
# meaning that we can't simply split train/test by sentence, as labels with small occurence might not end up in the test set
# the better way is to split the data by the occurence of each label, namely for each label there are 80% occurence in the train set and 20% in test
from collections import Counter

label_num = Counter()
total_labels = []

for _, annotations in prepared_data:
    for ent in annotations.get("entities"):
        total_labels.append(ent[2])
        label_num[ent[2]] += 1
        label_num['total'] += 1
total_labels = list(set(total_labels))
total_labels.sort()
print(f"Data label stat ({label_num['total']} in total):")
for label in total_labels:
    print(f"{label} count: {label_num[label]} ({round(label_num[label]/label_num['total']*100, 2)}%)")

# %%
# perform train-test split based on the occurence of each label according to the conclusion from last cell
import random

random.shuffle(prepared_data)

unfinished_lable = set(total_labels)

train_count = Counter()
test_count = Counter()

test_labels = []
test_data=[]
train_labels = []
train_data = []
for text, annotations in prepared_data:
        entity_labels = []
        for ent in annotations.get("entities"):
            entity_labels.append(ent[2])
        if set(entity_labels).issubset(unfinished_lable): # check if the current data's entities are still needed (not max out the 20% number for each class)
            # ent = (start_index,end_index,each_label) added
            for ent in annotations.get("entities"):
                test_labels.append(ent[2])
                test_count[ent[2]] += 1
                test_count['total'] += 1
            test_data.append((text, annotations))
            for label in list(unfinished_lable):
                if label in test_count and test_count[label] > label_num[label] * 0.2:
                    unfinished_lable.remove(label)
        else:
            for ent in annotations.get("entities"):
                train_labels.append(ent[2])
                train_count[ent[2]] += 1
                train_count['total'] += 1
            train_data.append((text, annotations))

# %% 
# Check the split test set stat
# Each label in test contains roughly 20% of their total occurence
# We can't split them into 20% precisely as we can't break a sentence,
# For example a sentence might carry a lot of entities that belong a rare label. 
# Whether this sentence is assigned to test or train set will significantly affect the percentage of such label in train/test set.
print()
print(f"Test set label stat ({test_count['total']} in total):")
test_labels = list(set(test_labels))
test_labels.sort()
for label in test_labels:
    print(f"{label} count: {test_count[label]} ({round(test_count[label]/label_num[label]*100, 2)}%) Total: {label_num[label]}")

# %%
# A quick check to see if the training data contains new labels that our spaCy model is not familiar with
# Result shows we need to add new labels to the model configuration in next cell
print()
print('All familiar labels in the spaCy model:',set(nlp.get_pipe('ner').labels))
print('All labels in the training set:',set(total_labels))
print('Whether spaCy model is familiar with all the labels in training set:',set(total_labels).issubset(set(nlp.get_pipe('ner').labels)))

# %%
# Add new labels in the training data to the NER model

# Getting the pipeline component
ner = nlp.get_pipe("ner")

for _, annotations in train_data:
    for ent in annotations.get("entities"):
        ner.add_label(ent[2])

# %%
# spaCy perform NER with a pipeline, we don't want the entire pipeline to be tuned
# so here we find the components that are not needed to be tuned
pipe_exceptions = ["ner", "trf_wordpiecer", "trf_tok2vec"]
unaffected_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]


# %%
# Train the model
from spacy.util import minibatch
from spacy.training.example import Example

# TRAINING THE MODEL
with nlp.disable_pipes(*unaffected_pipes):

    # Training for 30 iterations
    for iteration in range(30):
        print(f'Training iteration: {iteration}...')
    # shufling examples before every iteration
        random.shuffle(train_data)
        losses = {}
        # batch up the examples using spaCy's minibatch
        batches = minibatch(train_data, size=16)
        for batch in batches:
            example = []
            for text,annotations in batch:
                doc = nlp.make_doc(text)
                example.append(Example.from_dict(doc, annotations))
            nlp.update(example, 
                       drop=0.5,  # dropout - make it harder to memorise data
                       losses=losses,)

# %%
# Save the model
print()
print('Training finished, save model as "first_trained_model"')
nlp.to_disk("first_trained_model")

# %%
# Evaluate the result using test set
from utils import evaluate

print('Evaluate result for test data')
result = evaluate(nlp, test_data)

# %%
# Now we are going to enrich the entity LOC into GEO, GPE, STRL
# GEO : Geographic and natural features such as rivers, oceans, and deserts.
# GPE : Geopolitical Entity (GPE). Cities, countries/regions, states.
# STRL : Manmade structures.
# First make sure we have performed the "python term_extractor.py" so that "new_entity_terms.json" is saved to be used here
with open ("new_entity_terms.json",'r', encoding="utf-8") as f:  # added encoding method
    new_terms = json.load(f)

# %%
# Here we scrap the example sentence from wikipedia and label them with entities based on the term we have found in "new_entity_terms.json"
# The reason that we only collect the terms instead of sentence example previously is-
# We need our pre-trained model to join the sentence scrapping process,
# as we want to check first if the scrapped sentence does not have the label we want to enrich,
# because it can distract the model from learning the new labels in our second tuning.

# Note that this process is going to take some times as sentences have to be extracted from the web. and then getting labelled with entities.
from math import floor
from utils import data_creator # please check utils.py for more detail

# load our pre-trained spacy model to be further tuned for the new entity types, it is also used to label parts of the new found sentences
nlp = spacy.load('first_trained_model')

geo_terms = new_terms['GEO']
gpe_terms = new_terms['GPE']
strl_terms = new_terms['STRL']

# This part extracts the sentences and label them
print()
print('Extracting sentences from wikipedia to be used for enriching the label.')
print('This is going to take some times...')
geo_data = data_creator(nlp, geo_terms, 'GEO', 'LOC', n=1)
gpe_data = data_creator(nlp, gpe_terms, 'GPE', 'LOC', n=1)
strl_data = data_creator(nlp, strl_terms, 'STRL', 'LOC', n=1)

dataset = [geo_data, gpe_data, strl_data]
extra_train_data = []
extra_test_data = []

# train test splitting 8:2 for sentences for each new entity type
for data in dataset:
    random.shuffle(data)
    extra_train_data.extend(data[:int(floor(len(data) * 0.8))])
    extra_test_data.extend(data[int(floor(len(data) * 0.8)):])

random.shuffle(extra_train_data)
random.shuffle(extra_test_data)

#%% 
# save the webscrapped data 
print()
print('Saving the scrapped data as enrich_train_data.json, enrich_test_data.json')
with open('enrich_train_data.json', 'w', encoding='utf-8') as f:
    json.dump(extra_train_data, f, ensure_ascii=False, indent = 4)
with open('enrich_test_data.json', 'w', encoding='utf-8') as f:
    json.dump(extra_test_data, f, ensure_ascii=False, indent = 4)
    
# %%
# Turn the data into form suitable for spaCy training
from utils import transform_data

new_train_data = transform_data(extra_train_data)
new_test_data = transform_data(extra_test_data)

#%%
# set up the training
# Getting the pipeline component
ner = nlp.get_pipe("ner")

# add new entity types to the model
for _, annotations in new_train_data:
    for ent in annotations.get("entities"):
        ner.add_label(ent[2])

# Disable pipeline components you dont need to change
pipe_exceptions = ["ner", "trf_wordpiecer", "trf_tok2vec"]
unaffected_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]

#%%
# Second training: to enrich the entity types using the data we scrapped from the web
from spacy.training.example import Example

# use resume training for extra fine-tuning
optimizer = nlp.resume_training()  

# rehearse data is used for the model to not experience "catastrophic forgetting"
rehearse_text = []
for i in train_data:
    rehearse_text.append(i[0])

train_batch_size = 16
rehearse_batch_size = int(floor(len(rehearse_text)/(len(new_train_data)/train_batch_size)))

# TRAINING THE MODEL
with nlp.disable_pipes(*unaffected_pipes):

    # Training for 30 iterations
    print()
    for iteration in range(30):
        print(f'iteration: {iteration}...')

    # shufling examples before every iteration
        random.shuffle(new_train_data)
        random.shuffle(rehearse_text)
        losses = {}
        r_losses = {}
        # batch up the examples using spaCy's minibatch
        batches = minibatch(new_train_data, size=train_batch_size)
        rehearse_batches = minibatch(rehearse_text, size=rehearse_batch_size)
        for batch in batches:
            example = []
            for text,annotations in batch:
                doc = nlp.make_doc(text)
                example.append(Example.from_dict(doc, annotations))
            nlp.update(
                        example,  
                        drop=0.35,  # dropout - make it harder to memorise data
                        sgd=optimizer,
                        losses=losses)
            try:
                rehearse_batch = [Example.from_dict(nlp.make_doc(t), {}) for t in next(rehearse_batches)]
                nlp.rehearse(rehearse_batch, sgd=optimizer, losses=r_losses)
            except:
                print('rehearse batch run out during the training process, creating new rehearse batches')
                random.shuffle(rehearse_text)
                rehearse_batches = minibatch(rehearse_text, size=rehearse_batch_size)
                rehearse_batch = [Example.from_dict(nlp.make_doc(t), {}) for t in next(rehearse_batches)]
            nlp.rehearse(rehearse_batch, sgd=optimizer, losses=r_losses)
# %%
# Save the model
print()
print('Extra training (enriching one entity type) finished, save model as "second_trained_model"')
nlp.to_disk("second_trained_model")
# %%
# evaluate the model on extra test data
# the result shows the model does learn the new entity types while maintaining the knowledge of previously learned types
# the performence is largely depends on how the data is extract and labelled, also the training parameters
# more explaination is in README file
from utils import evaluate

print()
print('Note that for some labels such as "LOC", the score is 0 because the test data simply don"t have it.')
print('Evaluate result for test data use for enriching one entity type')

result = evaluate(nlp, new_test_data)
# %%
# evaluate the model on both previous test data and the extra test data

print()
print('Evaluate result for both first and second test data')

result = evaluate(nlp, test_data + new_test_data)