####
#Trains the fasttext word embedding model
####
import fasttext
import os

cwd = os.getcwd()

#relative, hardcoded path
path = "../../../../notes/"

print("Warning, this will take a long time.")
    
model = fasttext.train_unsupervised(path+"we_training_prepped.txt", dim = 300, epoch = 3, lr = 0.05, thread = 6, minCount = 200)

print("Model is trained")

#save the model
model.save_model("model42.bin")
