# Rasa Chatbot to perform Airbnb listings search and a booking flow.

### Conversational flows 
> https://github.com/sudha-vijayakumar/CMPE252_StayRec/blob/main/Conversational%20flows.pdf

### Demo
> https://www.youtube.com/watch?v=rpEfkagwHGE

### **Run Instructions:**

#### Pre-requisite 

**Environment**

  > Rasa Version      :         2.8.8
  
  > Minimum Compatible Version: 2.8.0
  
  > Rasa SDK Version  :         2.8.2
  
  > Rasa X Version    :         0.42.4
  
  > Python Version    :         3.8.8
  
  > Operating System  :         macOS-10.15.7-x86_64-i386-64bit
  
  > Python Path       :         /opt/anaconda3/bin/python

#### HOW TO RUN

**Install dependancies using** 
> pip install -r requirements.txt 

**Create feature vector embeddings using the below jupyter notebook,
(not checking-in files due to large file size > 100MB)**
> python3 createEmbedding.py

After completing this step, feature vectors will be created under the folder 'data/embeddings/lst_embeddings'.

**How to train?**
> rasa train 

**RUN RASA** 

**Terminal-1:**
> rasa shell

**Terminal-2:**
> rasa run actions
