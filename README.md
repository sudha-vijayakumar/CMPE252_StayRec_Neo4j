# Rasa Chatbot to perform Airbnb listings search and a booking flow.

Conversational flows: https://github.com/sudha-vijayakumar/CMPE252_StayRec/blob/main/Conversational%20flows.pdf

Demo: https://www.youtube.com/watch?v=rpEfkagwHGE

**Run Instructions:**

Pre-requisite: (not checking-in files due to large file size > 100MB)
Create feature vector embeddings,
https://github.com/sudha-vijayakumar/CMPE252_StayRec/blob/main/actions/Preprocessing.ipynb

After completing this step, feature vectors will be created under the folder 'data/embeddings/lst_embeddings'.

How to train?
> rasa train 

Terminal-1:
> rasa shell

Terminal-2:
> rasa run actions
