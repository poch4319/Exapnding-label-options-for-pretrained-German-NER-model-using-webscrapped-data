from utils import transform_data, training, evaluate
import json
import spacy
import argparse
import os

def main():
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", "-e", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch-size", "-b", type=int, default=16, help="Number of batch size")
    parser.add_argument("--drop-out", "-d", type=float, default=0.35, help="Drop out rate")
    args = parser.parse_args()

    # get train and test used in the first round of training
    # we will use them to prevent catastrophic forgetting
    with open ("processed_train_data.json",'r', encoding="utf-8") as f: 
        first_train_data = json.load(f)    
    with open ("processed_test_data.json",'r', encoding="utf-8") as f:  
        first_test_data = json.load(f)

    # get the web-scrapped data for this round of tuning
    with open ("enrich_train_data.json",'r', encoding="utf-8") as f:  
        extra_train_data = json.load(f)
    with open ("enrich_test_data.json",'r', encoding="utf-8") as f:  
        extra_test_data = json.load(f)
    new_train_data = transform_data(extra_train_data)
    new_test_data = transform_data(extra_test_data)

    # load the saved model from first training and train it on new data
    nlp = spacy.load('first_trained_model')
    training(nlp, new_train_data, args, rehearse_data=first_train_data)

    # Save the model
    print()
    print('Training finished, save model as "second_trained_model"')
    nlp.to_disk("second_trained_model")

    # Evaluate the result using new test set
    result = ['Result for new test data:']
    print()
    print('Evaluate result for new entity test data')
    print('Note that for some labels such as "LOC", the score is 0 because the test data simply don"t have it.')
    result.extend(evaluate(nlp, new_test_data))
    
    # Evaluate the result using combined new and old test set 
    result.append('')
    result.append('Result for new and old test data combined:')
    print()
    print('Evaluate result for test data combined from two rounds of training')
    result.extend(evaluate(nlp, (first_test_data + new_test_data)))

    # Save the evaluation result
    if not os.path.exists('./evaluation_result'):
        os.makedirs('./evaluation_result')
    print()
    print('Save the result in "evaluation_result" folder')
    with open('./evaluation_result/second_training.txt', 'w') as f:
        for line in result:
            f.write(line + '\n')
            
if __name__ == "__main__":
    main()
