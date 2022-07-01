from spacy.matcher import PhraseMatcher
from spacy.tokens import Span
from spacy.scorer import Scorer
from spacy.training.example import Example
from spacy.util import minibatch
from math import floor
from collections import Counter
import re
import random
import wikipediaapi
import json
import warnings

wiki_wiki = wikipediaapi.Wikipedia('de')

def sentence_extractor(nlp, text, term):
    """Extract all the sentences that contain the specific entity term from a giving text (wikipage text)

    Args:
        nlp : spacy NLP model
        text str: the whole wikipage text
        term str: the entity term we want to find from the wikipage

    Returns:
        dict: dictionary containing sentences that have our key word
        list: shuffled key list to the returned dictionary
    """
    matcher = PhraseMatcher(nlp.vocab)
    terms = [term]
    # Only run nlp.make_doc to speed things up
    patterns = [nlp.make_doc(text) for text in terms]
    matcher.add("TerminologyList", patterns)
    
    text = re.sub('\n', '', text)  # get rid of line break
    doc = nlp(text)
    matches = matcher(doc)
    
    sents = {}
    sents_key = set()

    for _, start, end in matches:
        span = doc[start:end]
        key = (span.sent.start, span.sent.end) #we use the start and end position of the sentence as the key for one sentence
        if key not in sents_key:
            sents_key.add(key)
            contents = {}
            contents['term'] = span.text
            contents['sentence'] = span.sent.text
            contents['term_char_idx'] = [(span.start_char - span.sent.start_char, span.end_char - span.sent.start_char)]
            contents['term_token_idx'] = [(span.start - span.sent.start, span.end - span.sent.start)]
            sents[key] = contents
        else: # when more than one same entity occur in a sentence
            assert sents[key]['sentence'] == span.sent.text
            sents[key]['term_char_idx'].append((span.start_char - span.sent.start_char, span.end_char - span.sent.start_char))
            sents[key]['term_token_idx'].append((span.start - span.sent.start, span.end - span.sent.start))
    
    sents_key = list(sents_key)
    random.shuffle(sents_key) # shuffle the data so we can draw randomly
    return sents, sents_key

def create_training_data(nlp, sents, sents_key, new_entity_type, original_entity_type, n=1):
    """Turn the extracted sentences into a saveable training data format which includes
    the sentence and labelled entities for each data.

    Args:
        nlp: spacy nlp model, which will be used to label the rest of the sentence besides the term with new entity type.
        sents (dict): extracted sentences returned from sentence_extractor()
        sents_key (list): key to the sents dict, which is also returned from the sentence extractor()
        new_entity_type (str): The new entity type that we want to labelled the term we've found (GEO, GPO, STRL in our case)
        original_entity_type (str): The entity type we want to enrich, which will be prevented from the returned data to prevent confusion during training.
        n (int, optional): For each of entity terms, the number of data to be produced. Defaults to 1.

    Returns:
        list: a list containing n data
    """
    output = []
    for key in sents_key:
        # put our entities in the sentence first
        example = sents[key]
        sent = example['sentence']
        sent_doc = nlp(sent)
        entity_spans = []
        try:
            for start, end in example['term_token_idx']:
                entity_spans.append(Span(sent_doc, start, end, new_entity_type))
                
        except:
            print(f'For sentence: "{sent}"')
            print(f'Tried finding the term "{example["term"]}" but failed.')
            print('This problem might result from the wrong sentence segmentation that spaCy performs.')
            print('Skip this sentence...')
            print('------')
            continue
        sent_doc.set_ents(entity_spans)
        
        # then let the model label the rest of the entities
        # the ones we labelled earlier won't be overwritten
        sent_doc = nlp(sent_doc)
        result = {}
        
        # now generate the data to be saved later
        result['data'] = sent
        result['annotations'] = []
        unusable = False
        for ent in sent_doc.ents:
            if ent.label_ == original_entity_type: # we dont want original entity type to exist in the training sentence, it will interfere the performance
                unusable = True
                break
            entity = {}
            entity['start'] = ent.start_char
            entity['end'] = ent.end_char
            entity['text'] = ent.text
            entity['labels'] = [ent.label_]
            result['annotations'].append(entity)
        if unusable:
            continue
        output.append(result)
        if len(output) == n:
            break
    return output

def data_creator(nlp, terms, entity_label, original_entity_type, n=1):
    """A functinon that wraps the sentence_extractor() and create_training_data() together to generate data for all the terms we've found

    Args:
        nlp: spacy nlp model
        terms (dict): terms that we found using term_extractor.py
        entity_label (str): The new entity type that we want to labelled the term we've found (GEO, GPO, STRL in our case)
        original_entity_type (str): The entity type we want to enrich, which will be prevented from the returned data to prevent confusion during training.
        n (int, optional): The amount of data to be generated for each term. Defaults to 1.

    Returns:
        list: a list containing data for all the terms, and is ready to be saved as a training data
    """
    print()
    print(f'Extracting data from wikipedia with entity type "{entity_label}" ...')
    result = []
    for term in terms:
        wiki_title = terms[term]
        page = wiki_wiki.page(wiki_title)
        if page.exists() == False:
            continue
        sents, sents_key = sentence_extractor(nlp, page.text, term)
        data = create_training_data(nlp, sents, sents_key, entity_label, original_entity_type, n)
        if len(data) == 0: # for sentence failed to extract any entity from
            continue
        result.extend(data)
    print(f'For {len(terms)} terms that could be lablled as {entity_label},')
    print(f'tried extracting {n} data per term, successfully collected {len(result)} data.')
    return result 

def transform_data(data):
    """Turn data load from json file into spaCy training format

    Args:
        data (list): data freshly load from json

    Returns:
        list: a list containing all the data ready to be put in spaCy training
    """
    prepared_data=[]
    for row in data:
        if row.get('annotations'):
            entities=[]
            for annotation in row.get('annotations'):
                start_index = annotation.get('start')
                end_index = annotation.get('end')
                labels = annotation.get('labels')
                for each_label in labels:
                    entities.append((start_index,end_index,each_label))
                    
            text = row.get('data')
            prepared_data.append((text,{"entities":entities}))
    return prepared_data

def evaluate(nlp, test_set):
    """Using spaCy Scorer to perform evaluation of trained NER on test set and print the result

    Args:
        nlp : spaCy nlp model
        test_set (list): list containing all the test data

    Returns:
        list: list containing the result that are ready to be saved into texual file
    """
    examples = []
    scorer = Scorer()
    for text, gold_annotations in test_set:
        labelled_text = nlp(text) 
        example = Example.from_dict(labelled_text, gold_annotations)
        examples.append(example)
    scores = scorer.score(examples)
    
    text_report = []
    print()
    report = f"Precision: {round(scores['ents_p'],2)}, Recall: {round(scores['ents_r'],2)}, F1: {round(scores['ents_f'],2)}"
    text_report.append(report)
    print(report)
    
    ents_per_type = scores['ents_per_type']
    keys = list(ents_per_type.keys())
    keys.sort()
    for key in keys:
        report = f"{key}: Precision: {round(ents_per_type[key]['p'],2)}, Recall: {round(ents_per_type[key]['r'],2)}, F1: {round(ents_per_type[key]['f'],2)}"
        print(report)
        text_report.append(report)
    return text_report

def training(nlp, train_data, args, rehearse_data=None):
    """Train the NER for the given spacy model

    Args:
        nlp : spaCy nlp model
        train_data (list): the training data
        args : argparse namespace containing training parameters
        rehearse_data (list, optional): 
            the training set from previous training round if the model has been trained before,
            this training set is to prevent model from "forgetting" the learned labels in the past
        
    """
    warnings.filterwarnings("ignore")
    ner = nlp.get_pipe("ner")

    # add new entity labels
    for _, annotations in train_data:
        for ent in annotations.get("entities"):
            ner.add_label(ent[2])
    
    # collect pipeline components that don't need training
    pipe_exceptions = ["ner", "trf_wordpiecer", "trf_tok2vec"]
    unaffected_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]
    
    # parameters
    train_batch_size = args.batch_size
    num_epoch = args.epochs
    drop_out_rate = args.drop_out
    
    # parameters for rehearsing old data to prevent catastrophic forgetting
    if rehearse_data:
        optimizer = nlp.resume_training()  
        rehearse_text = []
        for i in rehearse_data:
            rehearse_text.append(i[0])
        rehearse_batch_size = int(floor(len(rehearse_text)/(len(train_data)/train_batch_size)))
    
    # training
    with nlp.disable_pipes(*unaffected_pipes):

        # Training for 30 iterations
        for iteration in range(num_epoch):
            print(f'Training iteration: {iteration}...')
        # shufling examples before every iteration
            random.shuffle(train_data)
            losses = {}
            if rehearse_data:
                random.shuffle(rehearse_text)
                r_losses = {}
                rehearse_batches = minibatch(rehearse_text, size=rehearse_batch_size)
            # batch up the examples using spaCy's minibatch
            batches = minibatch(train_data, size=16)
            for batch in batches:
                example = []
                for text,annotations in batch:
                    doc = nlp.make_doc(text)
                    example.append(Example.from_dict(doc, annotations))
                if rehearse_data:
                    nlp.update(example, 
                            drop=drop_out_rate,  # dropout - make it harder to memorise data
                            losses=losses,
                            sgd=optimizer)
                    try:
                        rehearse_batch = [Example.from_dict(nlp.make_doc(t), {}) for t in next(rehearse_batches)]
                        nlp.rehearse(rehearse_batch, sgd=optimizer, losses=r_losses)
                    except:
                        print('rehearse batch run out during the training process, creating new rehearse batches')
                        random.shuffle(rehearse_text)
                        rehearse_batches = minibatch(rehearse_text, size=rehearse_batch_size)
                        rehearse_batch = [Example.from_dict(nlp.make_doc(t), {}) for t in next(rehearse_batches)]
                    nlp.rehearse(rehearse_batch, sgd=optimizer, losses=r_losses)
                else:
                    nlp.update(example, 
                            drop=drop_out_rate,  # dropout - make it harder to memorise data
                            losses=losses)

def find_median(List):
    """return the median value of a sorted_list
    its just a helper function for upsampling in get_training_data()
    """
    number_of_data = len(List)
    if number_of_data % 2 == 0:
        median = (List[(number_of_data//2)]+List[(number_of_data//2-1)])/2
    else:
        median = List[(number_of_data//2)]
    return median

def get_training_data(upsampling=False):
    """load the data.json file, transform the data into spaCy input format, perform train/test split
    train/test split is done by splitting the occurences of each label, not by the number of the data.
    Since we can't split a sentence into two just because it might create better train/test split on label occurences,
    the resulting train/test ratio for each label will not be completely 8:2.

    Args:
        upsampling (bool, optional): Perform upsampling on sentences that contain labels that rarely occurences.
            Defaults to False.

    Returns:
        list, list: list of training data, a list of test data. Both are shuffled.
    """
    
    with open ("data.json",'r', encoding="utf-8") as f:  
        data = json.load(f)
    prepared_data = transform_data(data)
    
    label_num = Counter()
    total_labels = []

    for _, annotations in prepared_data:
        for ent in annotations.get("entities"):
            total_labels.append(ent[2])
            label_num[ent[2]] += 1
            label_num['total'] += 1
    total_labels = list(set(total_labels))
    total_labels.sort()

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

    if upsampling:
        occurence = []
        for label in train_count:
            occurence.append(train_count[label])
        occurence.sort()
        upper_quartile = find_median(occurence[len(occurence)//2:])
        
        boost = {}
        for label in train_count:
            if round(upper_quartile/train_count[label]) > 1:
                boost[label] = round(upper_quartile/train_count[label])
                
        train = list(train_data)
        for d in train_data:
            boosting = False
            scale = []
            for ent in d[1]['entities']:
                if ent[2] in boost:
                    boosting = True
                    scale.append(boost[ent[2]])
            if boosting:
                for _ in range(round(sum(scale) / len(scale))):
                    train.append(d)
                for ent in d[1]['entities']:
                    train_count[ent[2]] += round(sum(scale) / len(scale))
        train_data = train

    random.shuffle(train_data)
    random.shuffle(test_data)
    return train_data, test_data