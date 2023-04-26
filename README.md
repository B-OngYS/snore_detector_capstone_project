# snore_detector_capstone_project
This respository contains Jupyter notebooks and models trained for snore detection

# Context

According to the Mayo Clinic, "[s]noring is the hoarse or harsh sound that occurs when air flows past relaxed tissues in your throat, causing the tissues to vibrate as you breathe. Nearly everyone snores now and then, but for some people it can be a chronic problem."

Snoring can be an indicator of sleep apnea where a person might stop breathing for a certain amount of time during sleep. This may lead to insufficient rest or in the long run, heart problems.

# Problem Statement

The main goal of the project is to build a classifcation model that will detect the prescence of snoring in a sound clip.

We will be using accuracy as our main metric.


# Proposed Methods and Models

We will be using the librosa library to convert sound clips of snorning and non-snoring sounds to features, mainly, the Mel-frequency cepstral coefficients (MFCCs). MFCC are widely used in Speech Recognition Tasks and other audio classification problems.

We will then use Classification models (Logistic Regression, K-Nearest Neighbor and Random Forest) to return a prediction.


# Data source and assumptions

The data set used is based on an article by Dr. Tareq Khan whose data set has been uploaded to Kaggle.

T. H. Khan, "A deep learning model for snoring detection and vibration notification using a smart wearable gadget," Electronics, vol. 8, no. 9, article. 987, ISSN 2079-9292, 2019.

https://www.kaggle.com/datasets/tareqkhanemu/snoring

We will be assuming that the data and labeling of the data are accurate.

## Data Description

Below is the data description for the kaggle data source.

"The dataset contains two folders - one for snoring and the other for non-snoring.

Folder 1 contains snoring sounds. It has total 500 sounds. Each sound is 1 second in duration.
Among the 500 snoring samples, 363 samples consist of snoring sounds of children, adult men and adult women without any background sound. The remaining 137 samples consist of snoring sounds having a background of non-snoring sounds.

Folder 0 contains non-snoring sounds. It has total 500 sounds. Each sound is 1 second in duration.
The 500 non-snoring samples consist of background sounds that might be available near the snorer. Ten categories of non-snoring sounds are collected, and each category has 50 samples. The ten categories are baby crying, the clock ticking, the door opened and closed, total silence and the minor sound of the vibration motor of the gadget, toilet flashing, siren of emergency vehicle, rain and thunderstorm, streetcar sounds, people talking, and background television news."

The one second clips are segments of longer sound clips and labelled by the data provider.

# Methodology and Results

In this project, MFCCs were extracted using the librosa library. Our contraints were of computational power and time.

MFCCs can be viewed as compressed summaries of the characteristics of a sound clip. This will help us improve the speed of modelling while keeping computational requirements low.

Depending on how well our models perform, we can choose the granularity of data by increasing or decreasing the number of MFCC features.

Using Logistic Regression as our base line model, we found that 16 features was able to get an accuracy of about 70% for both train and test data.

We found that 64 was the optimal number of features for Logistic Regression which had an accuracy of 94%.

After using GridCV to test multiple classification models, our best model is a Random Forest Model (RFM) with 1500 estimators getting a 98% accuracy score.

# Unseen Data Test, Model Tuning, and Model Deployment

The unseen data is a clip from a morning talk show (https://www.youtube.com/watch?v=f4U9ZpsjZZs) where the female host plays an audio clip of her co-host snoring.

The audio clip had laughter and talking on it and the length of the clip used was a little more than 45 seconds.

To use our model, we had to transform the clip into 46 one second clips (the last one second clip had to be padded).

Our RFM was only able to capture 4 out of 9 periods of snoring. To increase the sensitivity, we lowered the threshold probability to 0.4 to capture all 9 periods of snoring.

The final model was deployed onto Streamlit and the script is in the main repo as main.py.

It sums the number of seconds of snoring and shows the waveform graph of the sound clip.

# Future work

For future work, this model could be a submodule of a sleep apnea detection product. In addition to the total time of snoring, the app could also calculate the periodicity of snoring and also if there is an absence of breathing.

With more computational resources, we could try using a deep learning model to classify different types of snores.

With more audio expertise, we could also use the MFCCs to provide some explanation of how it identifies snoring.

# Special Thanks

I would like to thank the following for their support and training:

- The instruactional staff including Ryan Chang, Ming Jie Tan, and Jimmy at General Assembly for their training

- My wife, Bek Wuay Tang, for her support through this transition period from being a teacher to a computer scientist.

- The Singapore DSI-35 cohort at General Assembly for being such a fun bunch to hangout with and bounce ideas off.

# Contributions
If you would like to contribute to this project, please fork and submit a pull request. I am always open to feedback and would love help with this project.
