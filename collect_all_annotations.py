####
#Links annotations in xml-files to the notes in text-files. Also splits the text into sentences and transforms the span of the annotatins to sentence_id 
####
import os
import xml.etree.ElementTree as ET
from nltk.tokenize import sent_tokenize as ST
import nltk
import pickle

#comment if already downloaded
nltk.download("punkt")

#hardcoded input folders
directoryxml = "/data/homedirs/stan/code_val_prev/code_stan/Annotations/Xmlfiles/"
directorytxt = "/data/homedirs/stan/code_val_prev/code_stan/Annotations/"

#counters used to validate whether the code works well
counter100 = 0
counter300 = 0
counter500 = 0
counter1 = 0
counter = 0


lst = list()
ignored_files = list()
output = dict()
lstLongAnnot = list()


def find_sentence(sentence_list, taggedspan, filename):
    """
    Find the sentence index that corresponds with the span.

    Keyword arguments:
    sentence_list: list -- list of all sentences in the note (list)
    taggedspan: the span of the annotation (list)
    filename: the name of the file (string)
    """

    sent_nrs = list()

    #for each span in the list of taggedspan
    for span in taggedspan:
        	
        #if the first number of the span is larger than the length of the first sentence in the sentencelist, pop the sentence and
        #subtract the length of it from the span. This code uses this concept to iteratively find the sentence that matches the span.
        sentence = sentence_list.copy()
        amount_char = 0
        sent_nr = -1
        
        
        
        while amount_char <= int(span[0]):
            amount_char += len(sentence[0])
            sent_nr += 1
            sentence.pop(0)
        
        #if there are only 20 characters in a new sentence of a multi-sentence annotation annotated, it is a randomly annotated word, and hence ignored
        #This code could have been better using a recursive function, but I chose against it since it won't be needed when annotations are revised (that'll eliminate multi-sentence annotations)
        temp_length = int(span[1]) - int(span[0]) - len(sentence_list[sent_nr])
        if  temp_length > 20:
            if temp_length - len(sentence_list[sent_nr+1]) > 20:
                if temp_length - len(sentence_list[sent_nr+1]) - len(sentence_list[sent_nr+2]) > 20:
                    if temp_length - len(sentence_list[sent_nr+1]) - len(sentence_list[sent_nr+2]) -len(sentence_list[sent_nr+3]) > 20:
                        if temp_length - len(sentence_list[sent_nr+1]) - len(sentence_list[sent_nr+2]) -len(sentence_list[sent_nr+3]) - len(sentence_list[sent_nr + 4]) > 20:
                            if temp_length - len(sentence_list[sent_nr+1]) - len(sentence_list[sent_nr+2]) -len(sentence_list[sent_nr+3]) - len(sentence_list[sent_nr + 4]) - len(sentence_list[sent_nr + 5])> 20:
                                sent_nrs.append([sent_nr, sent_nr +1, sent_nr +2, sent_nr + 3, sent_nr + 4, sent_nr + 5, sent_nr + 6])
                                
                            else:
                                sent_nrs.append([sent_nr, sent_nr +1, sent_nr +2, sent_nr + 3, sent_nr + 4, sent_nr + 5])
                        else:
                            sent_nrs.append([sent_nr, sent_nr +1, sent_nr +2, sent_nr + 3, sent_nr + 4])
                    else:
                        sent_nrs.append([sent_nr, sent_nr +1, sent_nr +2, sent_nr + 3])
                else:
                   sent_nrs.append([sent_nr, sent_nr +1, sent_nr +2])
            else:
                sent_nrs.append([sent_nr, sent_nr +1])
            
           
        else:    
            sent_nrs.append(sent_nr)
    
    
        
    return sent_nrs
                    

#extract information from xml-files
for xmlfile in os.listdir(directoryxml):
    tree = ET.parse(directoryxml + xmlfile)
    root = tree.getroot()
        
    taggedspans = list()
    tags = list()
    ignore_file = False
        
    filename = root.attrib.get("textSource")
        
    for child in root:
        skip = False
        if child.tag == "annotation":
            text = child[3].text
            try:    
                if len(text) >= 100:
                    counter100 +=1
                if len(text) >= 300:
                    counter300 +=1
                    
                    #ignoring any annotation with 300+ char
                    skip = True
                    
                if len(text) >= 500:
                    counter500 +=1
                    
            except:
                print("Error occured when trying to find annotated text: " + filename + ". Make sure the names from the annotations and the text files correspond." )
            if skip == False:
                attributes = child[2].attrib
                taggedspans.append([attributes.get("start"), attributes.get("end")])
                    
        if skip == False:    
            if child.tag == "classMention":
                tags.append(child[0].attrib.get("id"))
        
        

    
        
    #open the textfile for each xmlfile
    with open(directorytxt + filename) as infile:
                
        text = infile.read()         
        sentences_list = ST(text)
        
        #find the sentence corresponding the tag
        try:
            sent_nrs = find_sentence(sentences_list, taggedspans, filename)
            lst.append(sent_nrs)
        except Exception as e:
                print("find_sentence did not work for: " + filename + ". With error: " + str(e) + ". Make sure the annotation span does not exceed the document span. This note will not be added to the dataset.")
                #print(taggedspans)
                #print(e)
                #print(len(sentences_list))
                #print("\n")
                continue

        #originally used to skip any annotations that had at least one sentence of length 500    
        for sent in sentences_list:
            if len(sent) >= 500:
                ignore_file = False
                
        if ignore_file == True:
            continue
            print("Error")
        else:
            #save the sentence list, the indices of the sentences corresponding the tag and the tag as value in dictionary
            output[filename] = [sentences_list,sent_nrs,tags]
            
        #debugging purposes, if more sentences are tagges than there are tags, something went wrong. Isn't the case in the data used. 
        if len(sent_nrs) != len(tags):
            if len(sent_nrs) != 0:
                print("Error occured with " + filename+ " more tags found than sentences. Note should therefore be ignored.")
                
                    
#debugging purposes                    
print(counter100)
print(counter300)  
print(counter500)

#save output
pickle.dump(output, open("FinalAnnotations.p", "wb"))

#save the annotations that were long to revise it later
with open("longAnnotations2.txt", "w") as infile:
    for item in lstLongAnnot:
        for i in item:
            infile.write(i + " \t")
        infile.write("\n")

#debugging purposes    
print(len(lstLongAnnot))
