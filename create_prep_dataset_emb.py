####
#Takes an input file and preprocesses it to train the Word Embedding model on.
####
from nltk.stem.snowball import DutchStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize


stop_words = set(stopwords.words("dutch"))
ds = DutchStemmer()

#relative path, hardcoded. can be changed, needs to lead to hardcoded filename: "we_training_data_big.txt"
path = "../../../../notes/"

counter_list= list()
prep_text = list()

def preprocess(text):
    """
    Split text in tokens, ignore non-alphanumerical tokens, stem the token and delete if in stop_words

    Keyword argument:
    text: the text to preprocess (string)
    """

    string = ""

    #tokenizes the text
    tokens = word_tokenize(text)

    #used to count the total amount of tokens
    counter_list.append(len(tokens))

    #empty list
    stemlist = []

    #stem non-alphanumerical sequences and add to list
    for token in tokens:
        if token.isalnum():
            stemlist.append(ds.stem(token))
    
    #if token not in stop_words, add it to empty string
    new_text = [w for w in stemlist if not w in stop_words]
    for i in new_text:
        string+= i + " "
    
    #return result
    return(string)



print("Collection of data started. This can take a while.")

#read in the training data file, preprocess lines iteratively
with open(path+"we_training_data_big.txt") as infile:
    for line in infile:
        prep_text.append(preprocess(line))
     

#show example    
print(prep_text[0:20])

print("Preprocessing is done. Output will be written to file.")

#save the result
with open(path+"we_training_prepped.txt", "w") as outfile:
    for i in prep_text:
        outfile.write(i)
        
#debugging purposes
counter = 0
for i in counter_list:
    counter += i
print(counter)
