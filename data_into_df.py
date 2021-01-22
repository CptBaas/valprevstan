####
#Takes input data, transforms into dataframe needed for the rest of the scripts. Computes sentence embeddings for each sentence. 
####
import fasttext as ft
import pandas as pd
import pickle
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from nltk.stem.snowball import DutchStemmer
#uncomment if feeded
#nltk.download("stopwords")
stop_words = set(stopwords.words("dutch"))

#load in the trained word embedding model
word_emb = ft.load_model("model42.bin")

ds = DutchStemmer()

annot_dict = pickle.load(open("FinalAnnotations.p", "rb"))
counter_nested = list()

#this function will be made better accesible through boolean once a final application is needed, for now, it is run several times with different pieces commented/uncommented
def preprocess(sent):
    """
    Split text in tokens, ignore non-alphanumerical tokens, stem the token and delete if in stop_words

    Keyword argument:
    text: the text to preprocess (string)
    """
    tokens = word_tokenize(sent)
    
    stemlist = []
    for token in tokens:
        #comment/uncomment depending on preprocessing needed
        #if token.isalnum():
        stemlist.append(ds.stem(token))
    
    #comment/uncomment depending on preprocessing needed
    #new_sent = [w for w in stemlist if not w in stop_words]
    sent = ""

    #comment/uncomment if needed, and do opposite to the second loop
    #for i in new_sent:
        #sent+= i + " "

    #see above
    for i in stemlist:
        sent += i + " "

    #comment/uncomment either one, depending on preprocessing needed
    #return(sent.lower())
    return(sent)


def convert_tags(tag):
    """
    Transforms the tags to binary tags w.r.t. location (in vs out) and injury (injury vs no_injury)

    Keyword argument:
    tag: the tag to be transformed (string)
    """
    if tag in ["in_injury","in_no_injury","in_unspec_injury"]:
        inout = "in"
    elif tag in ["out_injury","out_no_injury","out_unspec_injury"]:
        inout = "out"
    else:
        inout = "unspec"
                        
    if tag in ["out_injury","in_injury","unspec_injury"]:
        injury = "injury"
    elif tag in ["out_no_injury","in_no_injury","unspec_no_injury"]:
        injury = "no_injury"
    else:
        injury = "unspec"
                    
    fall = "Fall"
    
    return(tag,fall,inout,injury)

def transform(df):
    """
    Transforms the dataframe into a better useable dataframe

    Keyword argument:
    df: the dataframe to be transformed (DataFrame)
    """
    data = pd.DataFrame(columns = ["Filename","Sentence", "Embedding","Tag", "InOut", "Injury", "Fall"])

    #tranform dataframe to dictionary
    dct = df.to_dict(orient = "index")
    iterator = 0
    
    for key, dict_data in dct.items():
        
        #save values
        sentid = dict_data.get(1)
        tagid = dict_data.get(2)
        
        #iterate over the list of sentences, while keeping the indices saved (needed to link it to sentence_id as extracted by collect_all_annotations.py
        for idx, sent in enumerate(dict_data.get(0)):
            fall = ""
            sent = preprocess(sent)
            emb = word_emb.get_sentence_vector(sent)
            
            #if sentence_id corresponds to the index of the sentence in the loop, save its tags
            if idx in sentid:
                tags = tagid[sentid.index(idx)]
                tag, fall, inout, injury =  convert_tags(tags)
            #else, do the same for any nested id (which is the case if the annotation is multi-sentence)
            else:
                for sublist in sentid:
                    if isinstance(sublist, list):
                        if idx in sublist:
                            counter_nested.append(1)
                            tags = tagid[sentid.index(sublist)]
                            tag, fall, inout, injury =  convert_tags(tags)
            
            #if there is no fall, all tags are O
            if fall != "Fall":     
                tag = "O"
                fall = "O"
                inout = "O"
                injury = "O"

            #make new dataframe    
            data.loc[iterator] = [key, sent, emb, tag, inout, injury, fall]
            iterator +=1
                
    return(data)
    



df = pd.DataFrame.from_dict(annot_dict, orient= "index")

#first split 70% as training data, then split remaining 30% half into test and validation set
train, temp = train_test_split(df,train_size = 0.7, random_state = 314)
test, val = train_test_split(temp,train_size = 0.5, random_state = 314)

train = transform(train)
test = transform(test)
val = transform(val)
    
#save all datasets
pickle.dump(train,open("TrainStemNoneNewModel.p","wb"))
pickle.dump(test,open("TestStemNoneNewModel.p","wb"))
pickle.dump(val,open("ValStemNoneNewModel.p","wb"))

#debugging purposes
print(len(counter_nested))
