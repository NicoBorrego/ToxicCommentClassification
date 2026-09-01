# Toxic Comment Classification

## What is about?

This repo is focused on learning how to develop, optimize and evaluate a supervised machine learning model to treat Natural Language Processing and classify a toxic or non-toxic comment provided by the user.

In this version, toxicity is defined as insulting language. It is not intended to detect every form of offensive comunication.

Once the model is selected, I proceeded with an external validation with [Tweets Dataset](https://www.kaggle.com/datasets/ashwiniyer176/toxic-tweets-dataset) from Kaggle.

## Selected model

The exploratory/model selection was performed in the notebook, the selected one is LinearSVC from SVM, trained in train.py and exported for inference.

## External validation (validation.py)

Once I treated the NaN values, I made a sample of 100 toxic and 100 non-toxic tweets, but the model predicted 120 non-toxic and 80 toxic, this may be related to differences in vocabulary, writing style, formatting or other characteristics between training and external datasets.

Toxicity Distribution:

```
toxicity_classification
False    120
True      80
Name: count, dtype: int64
--------------------------------------------------
                                                 tweet toxicity_classification
95   #oitnb â ¤ who's   !! #moi #me ð    #tgif #nom...                   False 
15   the greetings are so incredible @user thank yo...                   False 
30                            Sgp still in bitches dms                    True 
158  @KimberlyyAye @Leslieeeixta I think I did my j...                    True 
128  ' August Alsina Deserved that &#1043323;&#1043...                   False 
115  @user when saying @user surname keep saying as...                   False 
69   since i finished most of my assignments i'll c...                   False 
170  I tell bitches I got that I.T.S ( IKE TURNER S...                    True 
174  #theshallows weak attempt to cash in on the ja...                   False 
45   wonderful dinner â ¤ï¸  #dinner #yum #thegaffo...                   False 
``` 

## Deployment

The application is deployed and accesible through a web interface hosted on Cloudfare. The backend is built with FastAPI, containerized using Docker and deployed on Render.

[Live demo](https://bold-block-32cb.nicolasborregogonzalez.workers.dev/)
