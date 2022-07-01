import spacy
import json
from utils import training, evaluate, get_training_data
import argparse
import os

def main():
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--upsampling", "-u", default=False, action="store_true")
    parser.add_argument("--epochs", "-e", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch-size", "-b", type=int, default=16, help="Number of batch size")
    parser.add_argument("--drop-out", "-d", type=float, default=0.5, help="Drop out rate")
    args = parser.parse_args()

    # load nlp model
    nlp = spacy.load("de_core_news_lg")

    # Collect Data
    if args.upsampling:
        print()
        print('Perform upsampling when preparing training data')
    train_data, test_data = get_training_data(upsampling=args.upsampling)

    # saved the data because second_training.py will use it
    # we can't call get_training_data() there because they have been shufflled and turned into the the format for spaCy
    with open ("processed_train_data.json",'w', encoding="utf-8") as f:  
        json.dump(train_data, f, ensure_ascii=False, indent = 4)
        
    with open ("processed_test_data.json",'w', encoding="utf-8") as f:  
        json.dump(test_data, f, ensure_ascii=False, indent = 4)
        
    # Train
    training(nlp, train_data, args)

    # Save the model
    print()
    print('Training finished, save model as "first_trained_model"')
    nlp.to_disk("first_trained_model")

    # Evaluate the result using test set
    print()
    print('Evaluate result for test data')
    result = evaluate(nlp, test_data)
    
    # Save the evaluation result
    if not os.path.exists('./evaluation_result'):
        os.makedirs('./evaluation_result')
    print()
    print('Save the result in "evaluation_result" folder')
    with open('./evaluation_result/first_training.txt', 'w') as f:
        for line in result:
            f.write(line + '\n')
            
if __name__ == "__main__":
    main()
