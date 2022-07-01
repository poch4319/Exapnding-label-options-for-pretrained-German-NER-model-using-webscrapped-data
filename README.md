
# Exapnding label options for pretrained German NER model using webscrapped data
Below is a brief run through of my approach and thoughts toward the three sub tasks.

### Task 1 Fine-tune SpaCy german NER model for political domain data
I decided to split the train and test data with 8:2 ratio.
The problem is that the occurences for each entity type is extremely imbalanced as shown below.
```bash
'total': 622, 'ORG': 182, 'LOC': 114, 'PER': 102, 'PRF': 87, 'PRD': 47, 'EVN': 19, 'TIM': 19, 'NAT': 18, 'DOC': 18, 'DAT': 16
```
The most naive way of splitting the data would be to simply randomly draw 20% of the data to the test set,
but since the some labels are so rare, and that they might all exist within a same sentence. This method might result in test set not having sentence containing certain label.

I ended up splitting the train/test set based on the occurences of each table, but since a sentence (or a data point) can't be broken into several data,
for labels that occur rarely, the train/test ratio will not be at exactly 8:2, but should be somewhat around that ratio.

### Task 2 Training with upsampling
Using the splitted train/test data and the parameters set below in usage section,
the model is able to learn the new entities existing in the given dataset with decent performance.
For labels that occur rarely in the data such as `ENV` or `DOC`, the performance is still not ideal.

I tried to use upsampling for data that contains rare labels, the result has shown zero to mild improvement.
I beleive this is up to the difficulty of the sentences that contain rare label assigned to the test set.
Since the example containing rare labels are so little, if the difficult ones got assigned to the test set, then the test set cannot properly reflects the model performances on the rare labels.
Ultimately, to tackle the problem, we still have to collect more data for rare labels so that model can perform well on them.


Additionally, upsampling does not really change the proportion of each labels. As mentioned earlier, labels often co-exist in the same data. If we resample the data that contains rare labels, the popular labels might also get resampled as well.
Besides, upsampling significantly increase the training time for very little improvement. This goes back to the suggestion provided earlier. We will have to tackle the problem through data engineering by collecting more data for rare labels that do not necessarily co-exist with popular labels. 

### Task 3 Webscrapping data from wikipedia to expand entity types in the NER model
This part is where I spent most of my time on. My goal is to enrich `LOC` label into `GEO` for geographical location such as river or mountain,
`STRL` for structural and manmade location such as tower, stadium or bridge, and `GPE` for geopolitical entity such as country or states. 

Although wikipedia have rich information for many entries, 
there lacks a organized information that can allow us to extract entity belonging to certain types automatically.
My first attempt is to extract all the links within a given wikipage as they often leads to other wikipages. And based on the information given in each page,
I will be able to determine the entity that the page is about belong to a certain entity type.
For example, within the page of 'Germany' in wikipedia, there might be also links for wikipages for 'Berlin' and 'Rhine River'.
I might be able to build something to determine 'Berlin' is a `GPE` entity and 'Rhine River' is a `GEO` entity and scrap data from there.


Thus, I built an function that extract the categories that a wikipage belongs to, 
and turn the categories into word vector and compare the similarity of the vector with word vectors that are built with `GEO` or `GPE` or `STRL` meanings,
to determine the entity type of the title of the wikipage. It turns out that the categories of wikipedia pages are very broad and messy. For example for page 'Berlin', one category is '1230s establishments in the Holy Roman Empire', 
or page 'Eiffel Tower' contains too much categories related to the nation France, making it hard for its word vector to be differentiated between `STRL` or `GPE`.

Eventually, I can only extract terms that I know for sure belong to certain entity type by scrapping through pages such as 'List of european country and its capitals' in wikipedia,
to generate sentences that contain certain new entity type.

The data labelling is also quite challenging, as we can't simply labelled the terms that belong to the new entity types in scrapped data,
we also need to label all the rest of the entities that could be recognized by our NER model or else the pipeline will lose the ability to recognize entities that it has learned to recognize in the past.
This means I have to labelled the old entities using the trained model from last step before feeding the data into training pipeline. 
Another problem also occurs here as the scrapped sentence often contains other entities that could be labelled as new entity types.
For example, when extracting example for `GPE` term 'Taipei', we might get a sentence that is 'Taipei is the captial of Taiwan', where 'Taiwan' is also a `GPE`,
but we are yet to have a model that can label Taiwan as `GPE`, meaning it will be labelled as `LOC` from the previous model instead.
The resulting data might confused the model as whether or not to label something as `GPE` or `LOC`. 
My solution eventually is to find sentence that do not contain other `LOC` entities when scrapping for data with new entities,
but I believe this approach will limit the quality of the data as these new labels often co-exist with entities that belong to `LOC` but currently we are not feeding such data to the model.

As the result, the model that are trained with web-scrapped data with new entities does learn to label the new entities, but there are still rooms for the improvement. 
I believe the solution is also better data engineering, for example, to be able to label entities `LOC` with proper new entities so that we can include sentence with the new entity term with other `LOC` entities.
Another approach is to improve the source of the data (the domain of the data specifically). For example, for `STRL` terms, I am scrapping it through a page that is about the structures that are undergoing renovations in Germany.
As many terms in the pages belongs to historical buildings that were destroyed in war, sentences containing these terms are often revolving around historical background, 
which might not be similar to sentences that are written for modern structures.

Additionally, if we train the spaCy model with new data, the spaCy model is likely to experience **catastrophic forgetting**.
With the use of training data from last task, I'm able to rehearse the model so that it can still perform well on the entities that it learned to recognize from the training in last round.
### Time Spent
I spent roughly three hours for the task 1 and 2. 
I spent the rest of time when I'm off work researching the way to collect quality data to for task 3. 
There are more potential improvements I could make for the project in terms of organizing the repository better,
experimenting more training configuration and designing better data extracting pipeline.
## Installation

Install the required packages with pip:

```bash
  pip install -r requirements.txt
```
And make sure the spaCy german model is downloaded:
```bash
  python -m spacy download de_core_news_lg
```


## Usage
There are **two ways** you can explore my project. Both yields the same results.

### First approach:
This approach is to go through the entire three tasks through `challenge.py` file,
as it provides detailed code and comments on how I approach three task **step-by-step**.
Note that it could be messy and lengthy as I put everything in one single file.

First, simply run:
```bash
  python term_extractor.py
```

And go to `challenge.py` to run code cell by cell, as I've provided more detailed comments throughout my codes.
There are also more codes about data exploration that support my thought process.


### Second Approach:
Follow the below three steps, this is a more organized approach as it is only involved with command lines and also allows arguments for customization.
### Step 1: Train the loaded spaCy model with provided dataset
This includes the first and second tasks. Splitting the dataset and then train the model. 
The trained model will be saved to a folder called `first_trained_model`. 
The evaluation will also be performed and saved to a 
folder called `evaluation_result`. For potential improvement of the training result, upsampling can be turned on using `-u`.

```bash
  python first_training.py

```

Optional arguments: 

| Parameter                 | Default       | Description   |	
| :------------------------ |:-------------:| :-------------|
| -u --upsampling	        | False         |Perform upsampling for labels that rarely occur in training data
| -e --epochs               | 30            |The number training epochs (iteration)
| -b --batch-size 	        | 16	        |The number data within each training batch
| -d --drop-out  		    | 0.5	        | The drop out rate

### Step 2: Extract terms that are suitable for enriching entity
This is to prepare for task three: enriching existing entities.
The goal is to enrich the entity `LOC` into geographical location `GEO`, geopolitical location `GPE` and structural location `STRL`.

First run the below command. This is going to look through several wikipedia pages and collect terms that are related to our new entities.
The result will be saved to `new_entity_terms.json`. Note that they are just terms, not actual sentence example.
```bash
  python term_extractor.py
```
Then run the below command to scrap data for the terms we have found.
The program will go to each terms' wikipedia page, find the sentences that contain the term,
label the term with new entities and label other parts of the sentences with previously saved NLP model to construct a data set that could enrich the existing model's entity type.
The scrapped data will be saved to `enrich_train_data.json` and `enrich_test_data.json`.

Note that this process will take a couple of minutes, and error might happen due to the connection or wikipedia API issue.
If we only sample one sentence per term, we expect to get roughly 200 data per new entity.
```bash
  python scrap_data.py
```
Optional arguments: 

| Parameter   | Default  | Description   |	
| :-----------|:--------:| :-------------|
| -n --number | 1        |Number of sentences to be extract per term

### Step 3: Train the model again with freshly scrapped data
Here we introduce the new entity types `GPE`, `GEO` and `STRL` to the model through training using data we scrapped in last step.
To prevent the model from experiencing **catastrophic forgetting**, we also use the training data in step 1 for model to rehearse on the old data.

The trained model will be saved to a folder called `second_trained_model`. 
The evaluation will also be performed and saved to a 
folder called `evaluation_result`.

```bash
  python second_training.py
```

| Parameter                 | Default       | Description   |	
| :------------------------ |:-------------:| :-------------|
| -e --epochs               | 30            |The number training epochs (iteration)
| -b --batch-size 	        | 16	        |The number data within each training batch
| -d --drop-out  		    | 0.35	        | The drop out rate
