# Job Tag Classifier - (Semi-Supervised Learning)
We take a job title and description and predict the tags associated with that job. We do this through Natural Language Processing (NLP) and a Convolutional Neural Network (CNN). This model will train itself to become more accurate as more jobs are added by training it when new data is added. The model uses feature hashing which allows it to learn new words in the input space. Here is the pipeline used:

![Kiku](refs/pipeline.png)


# The Data

Our data is manually tagged with each tag for each job having its own line. We handle this in our feature processing but it is an important note of how to handle unstrucuted data. Here is a is an example of the data:

![Kiku](refs/data_example.png)

# The Model

Our model is a convultional nerual network that is 8 layers deep. ...

![Kiku](refs/network.png)

# The Performence

![Kiku](refs/performence_metrics.png)
