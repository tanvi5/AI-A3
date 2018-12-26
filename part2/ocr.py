#!/usr/bin/env python2
'''

Note: We have assumed the train-text input to be a normal text file(without any pos tagging), with which accuracy for this program increases. Accuracy of our classifier decreases if bc.train is given as input, the reason being symbols like ,.'" always are surrounded by spaces, unlike sentences in real world. 
In this program two models are deisgned for simplified version of ocr:

1) Simplified: This model considers only the emission probabilities for predicting the current letter.
	Emission probability = (1-p)^matched * (p)^unmatched, where p = 0.35. We tried different values ranging from 0.05 to 0.45. And 0.35 gave the best result.
	We consider probaility to be ( (1-p)^matched * (p)^unmatched ), as p is the percentage noise in prections assumed.  
	We have given different weightage to '*' matched/unmatched and ' ' matched/unmatched,so that noise won't affect predictions.Weightage of matching star > weightage of matching space
	We tried different values for weightage, the best results were obtained for val1,val2 to be (.75,.55),(.50,.45),(.40,.40),(.65,.50).
	In this case, we also tried using (1-p1)^star_matched * (p1)^star_unmatched * (1-p2)^space_matched * (p2)^space_unmatched, however after trying several values of p1 and p2,
	we switched to using different valuess of weightage be adding 0.5 for every match of space, 1.5 for every match of star, 0.45 for unmatched space and 1.55 for unmatched star. 
	We also tried calculating the value of 'p' used in formula for calculating emission probability based on average unmatched characters (considering them as noise) 
	and adjusting the p value by raising it to power 1.041 (which gave best result for other tried out values), however, fixing the value of p to 0.35, gave better results. 
	In this approach, we tried out calulating p as ((Average number of pixels unmatched)/(25*14)**(1.041))

	
2) Viterbi: Viterbi uses bayes net where observed value of letter also depend on prior letter.
	For Viterbi algorithm, we have used result and backtrack as a matrix of rows equal to number of letters available in the image and columns as number of all possible letters
	For first letter, we have calculated result value as initial probability * emission probability
	For all later letter, we maximize over the probabilities calculated for previous letters with all emission and its respective transition to current state
	This maximum value is stored in backtrack array as track of previous letters found.
	Initial probaility is calculated using the training text file provided.
	Here initial probability is calculated based on frequency of letters in the text file occuring as the first letter of sentence.
	So, initial probaility of a letter = number of occurences of that letter in the first position of sentence/Total number of sentences
	Transition probability is calculated based upon the occurences of letter2 after letter1 and stored in transition_prob dictionary as transtion_prob[letter1][letter2].
	So, transition probaility = number of occurences of letter2 after letter1/number of transitions
	Wherever, a transition probability or initial probability does not exists, the total number of occurences for that particular tranition or inital is set to 0.1.
	Emission probailities calculated for Simplified model are used in Viterbi to further calculate posterior probabilities.

While reading data, we are considering set of words appended with spaces from the file, so that transition from letter to space is also considered. Also, we are storing the same words in all small letters, all capital letters and with first capital and rest all as small letters. The reason being, there is a possibility of sentences in the images having all small, all capital or first capital and rest small letters (There are some such files in the input).



'''
from __future__ import division
from PIL import Image, ImageDraw, ImageFont
import sys
from math import log


CHARACTER_WIDTH=14
CHARACTER_HEIGHT=25


#This method reads data from file name, and returs words and first letter of each sentence
def read_data(fname):
    exemplars = []
    exemplars2 = []
    first = []
    file = open(fname, 'r')
    for line in file:
        # append a space at the end of each word, as line.split() will split on space. And space after word is important for training data as transition probability of space after letter will also be calculated. 
        data = tuple([w+' ' for w in line.split()])
        #append only the first letter of each line, which will be used to calculate initial probability.
        first.append(line[0].upper())
        #Initially we were considering the bc.train for training, which contains pos, so we are eliminating alternate words if the sentence contains one of the part of speech        
        if 'det' in data or 'noun' in data or 'adv' in data or 'verb' in data or 'pron' in data:
            exemplars += data[0::2]
        #if frequent pos are not in sentence, then consider all words of sentence
        else:
            exemplars += data
    #considered 3 types: 1) all lower case letters, all upper case, and words with first upper case and remaining lower case letters to train
    for f in exemplars:
        exemplars2.append(f.lower())                    #all lower
        exemplars2.append(f[0].upper()+f[1:].lower())   #first upper and remaining lower
        exemplars2.append(f.upper())                    #all upper
    return (exemplars2,first)

#Load letter method - untouched
def load_letters(fname):
    im = Image.open(fname)
    px = im.load()
    (x_size, y_size) = im.size
    result = []
    for x_beg in range(0, int(x_size / CHARACTER_WIDTH) * CHARACTER_WIDTH, CHARACTER_WIDTH):
        result += [ [ "".join([ '*' if px[x, y] < 1 else ' ' for x in range(x_beg, x_beg+CHARACTER_WIDTH) ]) for y in range(0, CHARACTER_HEIGHT) ], ]
    return result

#This method calculates emission probabilities for a given image input and accordingly returns probable string
def simplified(test_letters,train_letters):
    global emission_prob
    val1 = 0.5
    val2 = 0.45
    final = ''
    p = 0.35
    for letter_no, test_img in enumerate(test_letters):
        probability_list = []
        for letter, train_img in train_letters.items():
            matched = 0
            unmatched = 0
            #We have given different weightage to '*' matched/unmatched and ' ' matched/unmatched, so that noise wont affect predictions
            #We tried different values for weightage, the best results were obtained for val1,val2 to be (.75,.55),(.50,.45),(.40,.40),(.65,.50)
            for v_no in range(25):
                for h_no in range(14):
                    #weightage of matching star > weightage of matching space
                    if test_img[v_no][h_no] == train_img[v_no][h_no] and test_img[v_no][h_no] == '*':
                        matched+=(2-val1)
                    elif test_img[v_no][h_no] == train_img[v_no][h_no] and test_img[v_no][h_no] == ' ':
                        matched+=val1
                    #weightage of star unmatched > weightage of space unmetched
                    elif test_img[v_no][h_no] != train_img[v_no][h_no] and test_img[v_no][h_no] == ' ':
                        unmatched+=val2
                    else:
                        unmatched+=(2-val2)
            probability_list.append([letter,(((p)**unmatched)*((1-p)**matched))])
            #Emission probability = (1-p)^matched * (p)^unmatched, where p = 0.35. We tried different values ranging from 0.5 to 0.45. And 0.35 gave the best result
            #We consider probaility to be (1-p)^matched * (p)^unmatched, p is the percentage noise in prections assumed. 
            if letter_no not in emission_prob.keys():
                emission_prob[letter_no] = {letter:(((p)**unmatched)*((1-p)**matched))}
            else:
                emission_prob[letter_no][letter] = (((p)**unmatched)*((1-p)**matched))
        #Select letter having maximum probablity from the list    
        final += max(probability_list, key=lambda x: x[1])[0] 
    return final

#Calculates transition and initial probabilities based on words in file and first letters of ecah sentence
def train(train_data_for_words,train_data_for_init_letters):
    transition_sums = {}
    initial_sums = {}
    global transition_prob
    global initial_prob
    global learning_letter_count
    #Calculate initial probabilities by counting number of occurences of letters as first letters in the sentence
    for letter in range(len(train_data_for_init_letters)):
        if letter not in initial_sums.keys():
            initial_sums[letter] = 1
        else:
            initial_sums[letter] += 1
    #Calculate transition probailities by counting occurences transition from letter 1 to letter 2
    for word_no in range(len(train_data_for_words)): 
        learning_letter_count += len(train_data_for_words[word_no])
        #to calculate transition probabilities, first calculate 
        for letter_no in range(len(train_data_for_words[word_no])-1):
            l1 = train_data_for_words[word_no][letter_no]
            l2 = train_data_for_words[word_no][letter_no+1]
            if l1 not in transition_sums.keys():
                transition_sums[l1] = {l2:1}
            elif l2 not in transition_sums[l1].keys():
                transition_sums[l1][l2] = 1
            else:
                transition_sums[l1][l2] += 1
    #If transition from a particular letter 1 to letter 2 does not exists, then add a transition probability, in dictionary with minimum probability
    for l1 in train_letters.keys():
        for l2 in train_letters.keys():
            if l1 not in transition_sums.keys():
                transition_sums[l1] = {l2:0.1}
            elif l2 not in transition_sums[l1].keys():
                transition_sums[l1][l2] = 0.1    
    #If initial probaility for a particular letter not exists, then add an initial probaility, in dictionary with minimum probability
    for letter in train_letters.keys():
        if letter not in initial_sums.keys():
            initial_sums[letter] = 0.1
            
    for key in transition_sums.keys():
        transition_prob[key] = {key2:transition_sums[key][key2]/(learning_letter_count-len(train_data_for_words)) for key2 in transition_sums[key].keys()}
    for key in initial_sums.keys():
        initial_prob[key] = initial_sums[key]/len(train_data_for_init_letters)


'''Viterbi uses bayes net where observed value of letter also depends on prior letter.
   For Viterbi algorithm, we have used result and backtrack as a matrix of rows equal to number of letters available in the image and columns as number of all possible letters
   For first letter, we have calculated result value as initial probability * emission probability
   For all later letter, we maximize over the probabilities calculated for previous letters with all emission and its respective transition to current state
   This maximum value is stored in backtrack array as track of previous letters found.'''
def hmm_viterbi(test_letters,train_letters):
       global initial_prob
       global transition_prob
       global emission_prob
       all_possible_letters = train_letters.keys()
       result = []
       backtrack = []
       for i in range(len(all_possible_letters)):
           result.append([])
           backtrack.append([])
           for j in range(len(test_letters)):
               result[i].append(0)
               backtrack[i].append(0)
       max_cost = 0
       list_costs = []
       for i, test_letter in enumerate(test_letters):
           for j, possible_letter in enumerate(all_possible_letters):
               if i: # if test letter is not the first letter, posterior probaility will be max(prior * transition * emission)
                   list_costs = []
                   for k,s_trans in enumerate(all_possible_letters):
                       list_costs.append(result[k][i-1] + log(transition_prob[s_trans][possible_letter]))
                   max_cost = max(list_costs)

                   result[j][i] = max_cost + log(emission_prob[i][possible_letter])

                   li = [x for x in backtrack[list_costs.index(max_cost)][i-1]]
                   li.append(list_costs.index(max_cost))
                   backtrack[j][i] = li
               else: # if test letter is the first letter, then prosterior probaility is (initial probaility * emission probaility)
                   out = log(initial_prob[possible_letter]) + (log(emission_prob[i][possible_letter]))
                   list_costs.append(out)
                   result[j][i] = out
                   backtrack[j][i] = [0]
       list_costs = []        
       for ind in range(len(all_possible_letters)):
           list_costs.append(result[ind][-1])
       max_cost = max(list_costs)
       path = [x for x in backtrack[list_costs.index(max_cost)][-1]]
       path.append(list_costs.index(max_cost))
       path = path[1:]
       final_path =[]
       for i in range(0, len(path)):
           final_path.append(list(all_possible_letters)[path[i]])
       sentence = ''
       for l in final_path:
           sentence += l
       return sentence

#Load training letters untouched
def load_training_letters(fname):
    TRAIN_LETTERS="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789(),.-!?\"' "
    letter_images = load_letters(fname)
    return { TRAIN_LETTERS[i]: letter_images[i] for i in range(0, len(TRAIN_LETTERS) ) }

#Calls appropriate function to predict sentence from image based on model
def solve(model, letters, train_letters):
        if model == "Simple":
            return simplified(letters,train_letters)
        elif model == "Viterbi":
            return hmm_viterbi(letters,train_letters)
        else:
            print("Unknown algo!")


def percent_match(sent1,sent2):
    match = 0
    for i in range(min(len(sent1),len(sent2))):
        if sent1[i] ==sent2[i]:
            match+=1
    return (100*match)/(min(len(sent1),len(sent2)))



# main program              
(train_img_fname, train_txt_fname, test_img_fname) = sys.argv[1:] #'courier-train.png',"train_text.txt",'test-0-0.png' #sys.argv[1:]
learning_letter_count = 0
initial_prob = {}
transition_prob = {}
emission_prob = {}
train_letters = load_training_letters(train_img_fname)
train_data_for_words,train_data_for_init_letters = read_data(train_txt_fname)
train(train_data_for_words,train_data_for_init_letters)

test_letters = load_letters(test_img_fname)
output = solve( 'Simple', test_letters,train_letters) 
print 'Simple:',output

output = solve( 'Viterbi', test_letters,train_letters) 
print 'Viterbi:',output

print 'Final answer:'
print output


