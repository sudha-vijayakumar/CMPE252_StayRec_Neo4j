# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/custom-actions


# This is a simple example for a custom action which utters "Hello World!"
#
# REFERENCES
# - https://medium.com/betacom/unsupervised-nlp-task-in-python-with-doc2vec-da1f7727857d
# - https://medium.com/betacom/building-a-rasa-chatbot-to-perform-listings-search-60cea9829e60
# - https://homes.cs.washington.edu/~msap/acl2020-commonsense/slides/02%20-%20knowledge%20in%20LMs.pdf


from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher

import pandas as pd
from gensim.models.doc2vec import Doc2Vec
from gensim.parsing.preprocessing import preprocess_string

# Load ML model
root = './data/'

model = Doc2Vec.load('/Users/sudhavijayakumar/Documents/CMPE252_StayRec/data/embeddings/lst_embeddings')

# Load dataset to get listings titles
df = pd.read_csv(root+'listings.csv.gz', sep=',', usecols = ['listing_url','picture_url','name','description','neighbourhood','property_type','bedrooms','bathrooms','amenities','price','review_scores_rating']) 

class ActionlistingsDetails(Action):

	def name(self) -> Text:
		return "action_listings_details"

	def run(self, dispatcher: CollectingDispatcher,
			tracker: Tracker,
			domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

		userMessage = tracker.latest_message['text']
		# use model to find the listings
		new_doc = preprocess_string(userMessage)
		test_doc_vector = model.infer_vector(new_doc)
		sims = model.docvecs.most_similar(positive = [test_doc_vector])		
		
		# Get first 5 matches
		for s in sims[:1]:
			picture = df['picture_url'].iloc[s[0]]
			listingss = df['listing_url'].iloc[s[0]]
			name = df['name'].iloc[s[0]]
			description = df['description'].iloc[s[0]]
			neighbourhood = df['neighbourhood'].iloc[s[0]]
			bedroom = df['bedrooms'].iloc[s[0]]
			bathroom = df['bathrooms'].iloc[s[0]]
			amenities = df['amenities'].iloc[s[0]]
			price = df['price'].iloc[s[0]]
			review_score_rating = df['review_scores_rating'].iloc[s[0]]

		botResponse = "Please find the top listing details:\nlink: "+str(listingss)+"\nTitle: "+str(name)+"\nDescription: "+str(description)+"\nNeighbourhood: "+str(neighbourhood)+"\nBedroom: "+str(bedroom)+"\nBathroom: "+str(bathroom)+"\nAmenities: "+str(amenities)+"\nPrice: "+str(price)+" per night\nRating: "+str(review_score_rating)+" on a scale of 5"
		
		dispatcher.utter_message(text=botResponse)
		dispatcher.utter_message(image=picture)

		return []

class ActionlistingsSearch(Action):

	def name(self) -> Text:
		return "action_listings_search"

	def run(self, dispatcher: CollectingDispatcher,
			tracker: Tracker,
			domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

		userMessage = tracker.latest_message['text']
		# use model to find the listings
		new_doc = preprocess_string(userMessage)
		test_doc_vector = model.infer_vector(new_doc)
		sims = model.docvecs.most_similar(positive = [test_doc_vector])		
		
		# Get first 5 matches
		listingss = [df['listing_url'].iloc[s[0]] for s in sims[:5]]

		botResponse = f"Here are the listing details: {listingss}.".replace('[','').replace(']','')
		
		dispatcher.utter_message(text=botResponse)

		return []

class ActionlistingsPics(Action):

	def name(self) -> Text:
		return "action_listings_pics"

	def run(self, dispatcher: CollectingDispatcher,
			tracker: Tracker,
			domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

		userMessage = tracker.latest_message['text']

		new_doc = preprocess_string(userMessage)
		test_doc_vector = model.infer_vector(new_doc)
		sims = model.docvecs.most_similar(positive = [test_doc_vector])		
		
		listingss = [df['picture_url'].iloc[s[0]] for s in sims[:1]]

		str = ''
		for lst in listingss:
			str = lst

		botResponse = f"Your requirement seems to match with the following listing:"
		
		dispatcher.utter_message(text=botResponse)
		dispatcher.utter_message(image=str)

		return []

class ActionlistingsBook(Action):

	def name(self) -> Text:
		return "action_book"

	def run(self, dispatcher: CollectingDispatcher,
			tracker: Tracker,
			domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

		botResponse = f"Booking confirmed under your account! Confirmation sent to email."
		
		dispatcher.utter_message(text=botResponse)

		return []

class ActionlistingsCancel(Action):

	def name(self) -> Text:
		return "action_cancel"

	def run(self, dispatcher: CollectingDispatcher,
			tracker: Tracker,
			domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

		botResponse = f"Booking cancelled!"
		
		dispatcher.utter_message(text=botResponse)

		return []

class ActionBookUnder(Action):

	def name(self) -> Text:
		return "action_book_under_name"

	def run(self, dispatcher: CollectingDispatcher,
			tracker: Tracker,
			domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

		userMessage = tracker.latest_message['text']
	
		botResponse = "This is your,"+userMessage+" . Please tell me if i can proceed with booking!"
		
		dispatcher.utter_message(text=botResponse)
		return []

class ActionIssueProcessing(Action):

	def name(self) -> Text:
		return "action_issue_processing"

	def run(self, dispatcher: CollectingDispatcher,
			tracker: Tracker,
			domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

		userMessage = tracker.latest_message['text']
	
		botResponse = "Your "+userMessage+". We'll work on reported issue and get back! Tracking id is:xxxx"
		
		dispatcher.utter_message(text=botResponse)
		return []

