from math import floor
from utils import data_creator # please check utils.py for more detail
import json
import spacy
import random
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--number", "-n", type=int, default=1, help="Number of sentences to be extract per term")
    args = parser.parse_args()

    # First make sure we have performed the "python term_extractor.py"
    # so that "new_entity_terms.json" is saved to be used here
    with open ("new_entity_terms.json",'r', encoding="utf-8") as f: 
        new_terms = json.load(f)
        
    # load our pre-trained spacy model to be further tuned for the new entity types, it is also used to label parts of the new found sentences
    nlp = spacy.load('first_trained_model')

    geo_terms = new_terms['GEO']
    gpe_terms = new_terms['GPE']
    strl_terms = new_terms['STRL']

    # This part extracts the sentences and label them
    print()
    print('Extracting sentences from wikipedia to be used for enriching the label.')
    print('This is going to take some times...')
    geo_data = data_creator(nlp, geo_terms, 'GEO', 'LOC', n=args.number)
    gpe_data = data_creator(nlp, gpe_terms, 'GPE', 'LOC', n=args.number)
    strl_data = data_creator(nlp, strl_terms, 'STRL', 'LOC', n=args.number)
    dataset = [geo_data, gpe_data, strl_data]

    # train test splitting 8:2 for sentences for each new entity type
    extra_train_data = []
    extra_test_data = []

    for data in dataset:
        random.shuffle(data)
        extra_train_data.extend(data[:int(floor(len(data) * 0.8))])
        extra_test_data.extend(data[int(floor(len(data) * 0.8)):])

    random.shuffle(extra_train_data)
    random.shuffle(extra_test_data)

    # save the data ready to enrich an entity type in second training
    print()
    print('Saving the scrapped data as enrich_train_data.json, enrich_test_data.json')
    with open('enrich_train_data.json', 'w', encoding='utf-8') as f:
        json.dump(extra_train_data, f, ensure_ascii=False, indent = 4)
    with open('enrich_test_data.json', 'w', encoding='utf-8') as f:
        json.dump(extra_test_data, f, ensure_ascii=False, indent = 4)

if __name__ == "__main__":
    main()
