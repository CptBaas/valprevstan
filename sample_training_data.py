####
#Iteratively saves data into a single file, to be used to train the Word Embedding model on.
####
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
    
nltk.download("punkt")    
    
path = "../../../../notes/" 
counter = 0 
errorlog = ""

listpaths = ["vumc/predictions/all_predictions_nov_14.csv","vumc/predictions/all_predictions_sept_28.csv", "amc/predictions/pred_valrisico_amc_2017_part1_notities.csv","amc/predictions/pred_valrisico_amc_2017_part2_notities.csv","amc/predictions/pred_valrisico_amc_2017_part3_notities.csv","amc/predictions/pred_valrisico_amc_2017_part4_notities.csv","amc/predictions/pred_valrisico_amc_2017_part5_notities.csv","amc/predictions/pred_valrisico_amc_2018_part1_notities.csv","amc/predictions/pred_valrisico_amc_2018_part2_notities.csv","amc/predictions/pred_valrisico_amc_2018_part3_notities.csv","amc/predictions/pred_valrisico_amc_2018_part4_notities.csv","amc/predictions/pred_valrisico_amc_2018_part5_notities.csv"]


with open(path + "we_training_data_big.txt", "w") as outfile:
    print("geopend")
    for pathend in listpaths:
        for chunk in pd.read_csv(path + pathend, chunksize = 10000):
            print(pathend)
            for notitie in chunk["notitie"]:
                    try:
                        text = word_tokenize(notitie.lower())
                        outfile.write(" ".join(text) + "\n")
                    except:
                        #triggers if row is NA or empty
                        errorlog += pathend + "  " +str(id) + "\n"
                        print("error")
                    
with open(path+ "training_data_error.txt", "w") as outfile:
    outfile.write(errorlog)
