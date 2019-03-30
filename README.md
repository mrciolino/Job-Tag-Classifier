# Job Tag Classifier - (Semi-Supervised Learning)
We take a job title and description and predict the tags associated with that job. We do this through Natural Language Processing (NLP) and a Convolutional Neural Network (CNN). This model will train itself to become more accurate as more jobs are added by training it when new data is added. The model uses feature hashing which allows it to learn new words in the input space. Here is the pipeline used:

![Kiku](refs/pipeline.png)

# Data Collection

Our data is manually tagged with each tag for each job having its own line. Since this crates lots of duplicate rows we must handle this in our feature processing step. Here is a is an example of the data:

![Kiku](refs/data_example.png)

# Feature Creation and Feature Processing

After we grab the data from our SQL database, we convert it into a pandas dataframe before creating our features and turning the text into numerical data. We perform feature creation by generating text features, part of speech features, and finally collect all the job tags for each job id to remove duplicate rows. We are now ready for feature processing. We try to reduce the number of words and therefore the number of variables for input into our model by reducing words down to their root form. We do this through cleaning, stripping, and stemming. We then use feature hashing so our input space can recognize new words it has not seen. We then add all the features we created and the text data into one dataframe and send it to our model.

![Kiku](refs/feature_creation_and_feature_processing.png)


# The Model

Our model is a CNN with LTSM that has an embedding layer, 2 convolutional & pooling layers, a LSTM layer, and 3 dense layers forming or neural network. Here is a representation of our network.

![Kiku](refs/network.png)

# The Performance

![Kiku](refs/performence_metrics.png)
