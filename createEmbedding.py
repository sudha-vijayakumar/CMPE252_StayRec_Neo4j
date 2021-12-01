import pandas as pd
import gensim
from gensim.parsing.preprocessing import preprocess_documents
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import string

listings = pd.read_csv('data/listings.csv.gz', sep=',')
listings.info()

# drop columns that have no value to our recommendation
listings = listings[['id','listing_url','name','description','neighborhood_overview','picture_url', 
'property_type','room_type','accommodates','bathrooms','bathrooms_text',                               
'bedrooms','beds','amenities','price','minimum_nights','maximum_nights','review_scores_rating',                         
'review_scores_accuracy','review_scores_cleanliness','review_scores_checkin',
'review_scores_communication','review_scores_location']]

listings.fillna('0', inplace=True)

listings.reset_index(drop = True, inplace = True)

def remove_punc(sample_str):
    # Create translation table in which special charcters
    # are mapped to empty string
    translation_table = str.maketrans('', '', string.punctuation)
    # Remove special characters from the string using translation table
    sample_str = sample_str.translate(translation_table)
    return sample_str

listings['words_features'] = listings['amenities'].apply(remove_punc)

for ind in listings.index:
     listings['review_scores_rating'][ind] = (float(listings['review_scores_rating'][ind]) + float(listings['review_scores_accuracy'][ind]) + float(listings['review_scores_cleanliness'][ind]) + float(listings['review_scores_checkin'][ind]) + float(listings['review_scores_communication'][ind]) + float(listings['review_scores_location'][ind]))
     listings['review_scores_rating'][ind] = (listings['review_scores_rating'][ind])/6
     listings['words_features'][ind] = 'amenities:'+listings['words_features'][ind] +'description:'+  listings['description'][ind] +'neighborhood_overview:'+  listings['neighborhood_overview'][ind]+'property_type:'+  listings['property_type'][ind]+'room_type:'+  listings['room_type'][ind]+'accommodates:'+  str(listings['accommodates'][ind])+'bedrooms:'+  str(listings['bedrooms'][ind])+'beds:'+  str(listings['beds'][ind])+'price range:'+  listings['price'][ind]
listingss = listings.rename(columns={"review_scores_rating": "overall_rating"})
text_corpus = listingss['words_features'].values
processed_corpus = preprocess_documents(text_corpus)
tagged_corpus = [TaggedDocument(d, [i]) for i, d in enumerate(processed_corpus)]
model = Doc2Vec(tagged_corpus, dm=0, vector_size=200, window=2, min_count=1, epochs=100, hs=1)
model1= model
model.save('data/embeddings/lst_embeddings')
new_doc = gensim.parsing.preprocessing.preprocess_string("private room dishwasher")
test_doc_vector = model1.infer_vector(new_doc)
sims = model.docvecs.most_similar(positive = [test_doc_vector])
for s in sims:
    print(f"{(s[1])} | {listings['listing_url'].iloc[s[0]]}")