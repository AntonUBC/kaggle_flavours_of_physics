## My solution to [Kaggle Flavours of Physics challenge] (https://www.kaggle.com/c/flavours-of-physics)

This is a slightly improved version of my submitted solution (weighted ROC AUC score 0.994894 and the 9th place out of 673 teams).
This version gives a ROC AUC score 0.995177 and the 8th place on the leaderboard.

### Project Description 

Since the [Standard Model] (https://en.wikipedia.org/wiki/Standard_Model) of particle physics cannot explain some
observed physical phenomena, physicist are trying to develop extended theories, commonly labelled as physics
beyond the Standard Model or "[new physics](https://en.wikipedia.org/wiki/Physics_beyond_the_Standard_Model)". A clear sign of new physics would be the discovery of charged lepton flavour
violation, a phenomenon which cannot occur according to the Standard Model. The goal of the challenge was to help physicists to discover this phenomenon by developing a machine learning model which would maximize the discriminating power between 
signal events (where the violation did occur) and background events (where it did not). 

In this project we used real data from the [LHCb experiment] (http://lhcb-public.web.cern.ch/lhcb-public/)
at the [Large Hadron Collider](http://home.cern/topics/large-hadron-collider) (Switzerland).
Since the searched phenomenon has not been discovered yet, the real background data was mixed with simulated datasets
of the decay. This introduced two additional challenges into a modeling process:

- Since the classifier was trained on simulation data for the signal and real
  data for the background, it was possible to distinguish between signal and background simply
  by picking features that are not perfectly modeled in the simulation. To check our models for the presence of [this pattern](https://www.kaggle.com/c/flavours-of-physics/details/agreement-test)
  we were provided with additional data (both real and simulated) on the observed and well studied decay with similar characteristics.
  The Kolmogorov–Smirnov (KS) test was used to evaluate the differences between the classifier distributions on both
  samples of the observed decay. The submission requirement was that the KS-value of the test to be smaller than 0.09.
  
- Another requirement was that the classifier must be [uncorrelated with the tau-lepton mass](https://www.kaggle.com/c/flavours-of-physics/details/correlation-test), as
  such correlations can cause an artificial signal-like mass peak or lead to incorrect
  background estimations. In other words, since by definition the searched decay can occur only within a certain tau-mass range, a classifier could perform very well by simply assigning high probabilities to all observations within this range. However, such classifier would not be useful since this range can contain a large number of background observations as well. 
We performed the Cramer–von Mises (CvM) test to ensure the absence of such correlation. The CvM-value of the test was required to be smaller than 0.002.
  
### Model Description

The main challenge of this competition was developing a model which maximizes the probability of detecting
the decay and at the sane time passes KS and CvM tests. 

To address the first requirement, I used experimentation and removed some not perfectly
modeled features from data trying not to affect the performance of the classifier at the same time.

Addressing the second
requirement was much harder since the signal can occur only within a certain region of values of the tau-mass
and therefore is correlated with mass by definition. This means that any aggressive classifier which objective is
pure maximization of the ROC AUC score would eventually be correlated with the tau-mass. My solution to this problem was
twofold:

- I used the Uniform Gradient Boosting Classifier from the python library
  [hep_ml](https://arogozhnikov.github.io/hep_ml/) which is designed specifically for high energy physics. This classifier is
  able to maximize the objective function and at the same time minimize the correlation of predictions with certain specified variables.

- my model was constructed as an ensemble of two models: the first one was designed primarily for ROC AUC maximization while
  the goal of the second model was fighting the correlation with the tau-mass. I obtained my final predictions by finding      the optimal weights for ensembling these models.

Specifically, the first model was constructed using a three-level learning architecture.
At the first level I used the [stacked generalization algorithm](http://machine-learning.martinsewell.com/ensembles/stacking/)
to construct meta-features for both train and test data using signal predictions of 15 different models:

- KNeighbor Classifier (sklearn) with 5, 10, 20, 40, 80, 160, 320 neighbors
- Logistic Regression Classifier (sklearn)
- Gaussian Naive Bayes Classifier (sklearn)
- Support Vector Classifier (sklearn)
- Random Forest Classifier (sklearn)
- Extremely Randomized Trees Classifier (sklearn)
- Gradient Boosting Trees Classifier (sklearn)
- SGD Classifier (sklearn)
- Gradient Boosting Trees Classifier (XGBoost)

At the second level I trained XGBoost (XGB), Random Forest (RF) and Uniform Gradient Boosting (UGB) classifiers
on the original data combined with meta-features to obtain three different vectors of predicted probabilities. 

Finally, I combined obtained predictions using the following formula:

   ```preds_model1 = 0.3*(preds_xgb^0.65 * preds_rf^0.35) + 0.7*preds_ugb```
   
I found the optimal weights for this ensemble by using a hyper-parameter
search procedure of the python library [hyperopt](https://github.com/hyperopt/hyperopt). The 3-fold local CV score of this model was 0.994635.

The second model consisted of a single XGBoost classifier which was purposely "undertrained" by using a shallow tree structure and a relatively high learning rate. The 3-fold local CV score of this model was 0.984374.

The final vector of predicted probabilities was obtained by combining predictions of two models in the following way:

  ```preds_ensemble = preds_model1^0.585 * preds_model2^0.415```

### Instruction

- download train and test data from the [competition website](https://www.kaggle.com/c/flavours-of-physics) and put all the data
into folder ```./data``` (you may need to adjusts its path according to your location using ```/kaggle_flavours_of_physics//flavours_utils/paths.py```). You must also create a folder ```./submission``` in the same subfolder. This folder
will be used for saving predictions.

- run ```/kaggle_flavours_of_physics/ensemble/ensemble_submission.py``` to generate the file of predictions in csv format.   The stacking procedure used in the first model is computationally intensive and may take up to 12 hours to complete (it took me 11 hours on my 4-core 2.60GHz laptop with 16 GB RAM).
 
### Dependencies
- Python 3.4 (Python 2.7 would also work, just type: ```from __future__ import print_function``` in the beginning of the script)
- Pandas (any relatively recent version would work)
- Numpy (any relatively recent version would work)
- Sklearn (any relatively recent version would work)
- Hep_ml 0.3.0
- XGBoost 0.4.0
