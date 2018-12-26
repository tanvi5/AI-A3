#!/usr/bin/env python3

###################################
# CS B551 Fall 2018, Assignment #3
#
# shsowani-tpatil
#
#
####
# Put your report here!!
'''

Part of speech tagger uses simple, HMM viterbi and MCMC models to predict the part of speech 

While training data, we calculated initial_prob dictionary which will store all the probabilities of each of part of speech based on training file. 
transition_prob is a dictionary of dictionary storing transition probability from one part of speech to other
transition_prob2 is a dictionary of dictionary of dictionary storing transition probability from one part of speech to next to next which is used to calculate P(s3/s1,s2)
emission_prob is a dictionary storing probability of word given part of speech based on training file. 


Simple model- 
Simple model uses Simple bayes net where the observed variable word depends only on the unobserved variable part of speech
We calculate P(pos/word) = P(word/pos) * P(pos) as we are maximizing over the probabilities, we are neglecting denominator part here

HMM Viterbi - 
 Viterby uses bayes net where observed value word also depends on part of speech of prior word.
    For Viterbi algorithm, we have used result and backtrack as a matrix of rows equal to number of Part of speech available and columns as number of words in sentence
    For first word, we have calculated result value as initial probability * emission probability 
    For all later wods, we maximize over the probabilities calculated in previous word and its transition to current state
    This maximum value is stored in backtrack array as a containing a track as a list of part of speech found.
    
    
Complex MCMC - 

This is main MCMC function which will use sampling performed by generate_samples() function 
    We are generating 700 samples and discarding first 200 to improve sampling accuracy
    All the samples after 200 samples are stored in a list and then voting is taken for each of the word. 
    This idea of voting is given by Nithish . 
    Generate sample function is used to generate sample.
        
    This function will take each word at a time and find probability of pos tag given given all the other words and their corresponding tags
    that is we are calculating P(s1/w1, s2,s3..sn,w2,w3,w4,..wn) for every word in sentence. 
    If the test word is not in transition or emission probability, we are assuming some small value 1/10^-6 
    

To calculate Posterior Probabilities - 
     P(S1,S2,...,Sn | W1,W2,...,Wn)
For simple model, we are considering that the observed word only depends upon its pos tag
Here, 
     P(S1/W1) = P(W1/S1) * P(S1) for each word by neglecting denominator

For Complex MCMC model, 
Here we are considering 2 prior transitions as well. 
P(S1,S2,..Sn /W1, W2,.. Wn) = P(W1/S1) * P(W1) * P(W2/S2) * P(W2) * P(S2/S1) * P(W3/S3) * P(W3) * P(S3/S1,S2) .. and so on

For HMM Viterbi algorithm, 
Here we are only considering one prior transition 
P(S1,S2,..Sn /W1, W2,.. Wn) = P(W1/S1) * P(W1) * P(W2/S2) * P(W2) * P(S2/S1).. and so on



Assumptions - 
If test data set had new unseen word in emission or transition we are considering a very small value of probability as 1/10^-6

Final Results - 
==> So far scored 2000 sentences with 29442 words.
                   Words correct:     Sentences correct: 
   0. Ground truth:      100.00%              100.00%
         1. Simple:       93.92%               47.45%
            2. HMM:       95.03%               54.05%
        3. Complex:       90.53%               34.40%
        

'''
####

import random
import math
import re
from math import log
import numpy as np
from collections import Counter
# We've set up a suggested code structure, but feel free to change it. Just
# make sure your code still works with the label.py and pos_scorer.py code
# that we've supplied.
#
class Solver:
    
    # Global variables to store probabilities
    initial_prob = {}
    transition_prob = {}
    transition_prob2 = {}
    emission_prob = {}
    pos_prob = {}
    
    
    # Calculate the log of the posterior probability of a given sentence
    #  with a given part-of-speech labeling. Right now just returns -999 -- fix this!   
    '''
    To calculate 
     P(S1,S2,...,Sn | W1,W2,...,Wn)
    '''
    def posterior(self, model, sentence, label):

        if model == "Simple":
            out = 0
            for i,s in enumerate(label):
                s = label[i]
                word = sentence[i]
                emission =self.emission_prob[s][word] if s in self.emission_prob.keys() and word in self.emission_prob[s].keys() else 0.0000001
                
                out += log(emission) 
                out += log(self.pos_prob[label[i]]) if label[i] in self.pos_prob.keys() else log(0.0000001)
            return out
        
        elif model == "Complex":
            out = 0       
            for index in range(len(label)):
                s = label[index]
                word = sentence[index]
                emission =self.emission_prob[s][word] if s in self.emission_prob.keys() and word in self.emission_prob[s].keys() else 0.0000001
                
                if(index == 0):
                    out += log(self.initial_prob[s]) + log(emission)
                elif(index==1):
                    trans1 = self.transition_prob[s][label[index-1]] if s in self.transition_prob and label[index-1] in self.transition_prob[s].keys() else 0.0000001
                    out += log(emission) + log(trans1)
                else:
                    trans1 = self.transition_prob[s][label[index-1]] if s in self.transition_prob and label[index-1] in self.transition_prob[s].keys() else 0.0000001
                    trans2 = self.transition_prob2[s][label[index-1]][label[index-2]] if s in self.transition_prob2 and label[index-1] in self.transition_prob2[s].keys() and label[index-2] in self.transition_prob2[s][label[index-1]].keys()  else 0.0000001
                    out += log(emission) + log(trans2)
                    
            
            return out
        
        elif model == "HMM":
            out = 0
            for i in range(len(label)):
                s = label[i]
                word = sentence[i]
                emission =self.emission_prob[s][word] if s in self.emission_prob.keys() and word in self.emission_prob[s].keys() else 0.0000001
                
                if(i == 0):
                    out += log(self.initial_prob[s]) + log(emission)
                else:  
                    trans1 = self.transition_prob[s][label[i-1]] if s in self.transition_prob and label[i-1] in self.transition_prob[s].keys() else 0.0000001
                    out += log(trans1) + log(emission)
            
            return out
        
        else:
            print("Unknown algo!")

    # Do the training!
    #
    
    
    def train(self, data):
       transition_sums = {}
       transition_sums2 = {}
       initial_sums = {}
       emission_sums = {}
       pos_sums ={}
       for sentence_no in range(len(data)):
           first_word = data[sentence_no][1][0]
           if first_word not in initial_sums.keys():
               initial_sums[first_word] = 1
           else:
               initial_sums[first_word] += 1
           for pos in data[sentence_no][1]:
               if pos not in pos_sums:
                   pos_sums[pos] = 1
               else:
                   pos_sums[pos] = pos_sums.get(pos)+1
    
           
           for word_no in range(len(data[sentence_no][1])):
               
               if(word_no < len(data[sentence_no][1])-1):      
                   pos1 = data[sentence_no][1][word_no]
                   pos2 = data[sentence_no][1][word_no+1]   # Transition 1
                   
                   if pos1 not in transition_sums.keys():
                       transition_sums[pos1] = {pos2:1}
                   elif pos2 not in transition_sums[pos1].keys():
                       transition_sums[pos1][pos2] = 1
                   else:
                       transition_sums[pos1][pos2] += 1
                       
               if(word_no < len(data[sentence_no][1])-2):      
                   pos3 = data[sentence_no][1][word_no+2]   # Transition 2
                   if pos1 not in transition_sums2.keys():
                       transition_sums2[pos1] = {pos2:{pos3: 1}}
                   elif pos2 not in transition_sums2[pos1].keys():
                       transition_sums2[pos1][pos2] = {pos3:1}
                   elif pos3 not in transition_sums2[pos1][pos2].keys():
                       transition_sums2[pos1][pos2][pos3] = 1
                   else:
                       transition_sums2[pos1][pos2][pos3] += 1
            
               # FInd emission by count of words in each POS
               word = data[sentence_no][0][word_no]
               pos = data[sentence_no][1][word_no]
               if pos not in emission_sums.keys():
                   emission_sums[pos] = {word:1}
               elif word not in emission_sums[pos].keys():
                   emission_sums[pos][word] = 1
               else:
                   emission_sums[pos][word] += 1
    
       # Find probabilities from count 
       total = sum(pos_sums.values())
       for pos in pos_sums.keys():
           self.pos_prob[pos] = pos_sums[pos]/total
           
       for key in initial_sums.keys():
           self.initial_prob[key] = initial_sums[key]/len(data)
           
       for key in transition_sums.keys():
           total = sum(transition_sums[key].values())
           for key_new in transition_sums[key]:
                transition_sums[key][key_new] = transition_sums[key][key_new]/total
                
       for key1 in transition_sums2.keys():
            for key2 in transition_sums2[key1].keys():
                total = sum(transition_sums2[key1][key2].values())
                for key_new in transition_sums2[key1][key2]:
                    transition_sums2[key1][key2][key_new] = transition_sums2[key1][key2][key_new]/total
           
           
       for key in emission_sums.keys():
           total = sum(emission_sums[key].values())
           for key_new in emission_sums[key]:
                emission_sums[key][key_new] /= total
           
           
       self.transition_prob = transition_sums
       self.emission_prob = emission_sums
       self.transition_prob2 = transition_sums2
           
       
       '''
       For Simplified, for every word, we maximized over P(pos)*P(word | pos)     
       '''
    def simplified(self, sentence):
        pos = self.initial_prob.keys()
        probabilies_by_label = {}
        predicted_pos = []
        for word in sentence:            
            for s in pos:     
                prob = 1
                try:
                    prob = prob * self.emission_prob[s][word]
                except:
                    prob = prob * 0.000001
                probabilies_by_label[s] = prob * self.pos_prob[s]   # POS prior probability * emission probability
            out = max(probabilies_by_label, key = probabilies_by_label.get)
            predicted_pos.append(out)
        
        return predicted_pos
    
    '''
    Generate sample function is used to generate sample.        
    This function will take each word at a time and find probability of pos tag given given all the other words and their corresponding tags
    that is we are calculating P(s1/w1, s2,s3..sn,w2,w3,w4,..wn) for every word in sentence. 
    If the test word is not in transition or emission probability, we are assuming some small value 1/10^-6 
    
    '''
    def generate_samples(self,samples, sentence):
        pos = self.initial_prob.keys()
        for index, word in enumerate(sentence):
            probabilities = []
            for s in pos:
                # calculate probabilities
                emission =self.emission_prob[s][word] if s in self.emission_prob.keys() and word in self.emission_prob[s].keys() else 0.0000001
                
                if(index == 0):
                    probabilities.append(self.initial_prob[s] * emission)
                elif(index==1):
                    trans1 = self.transition_prob[s][samples[index-1]] if s in self.transition_prob and samples[index-1] in self.transition_prob[s].keys() else 0.0000001
                    probabilities.append(emission * trans1)
                else:
                    trans2 = self.transition_prob2[s][samples[index-1]][samples[index-2]] if s in self.transition_prob2 and samples[index-1] in self.transition_prob2[s].keys() and samples[index-2] in self.transition_prob2[s][samples[index-1]].keys()  else 0.0000001
                    probabilities.append(emission *  trans2)
                    
            new_list = [x/sum(probabilities) for x in probabilities]
            cdf = 0
            guess = random.random()
            for count, val in enumerate(new_list):
                cdf += val
                if guess<cdf:
                    samples[index] = list(pos)[count]
                    break
                    
        return samples
    '''
    This is main MCMC function which will use sampling performed by generate_samples() function 
    We are generating 700 samples and discarding first 200 to improve sampling accuracy
    All the samples after 200 samples are stored in a list and then voting is taken for each of the word. 
    This idea of voting is given by Nithish . 
    
    '''
    def complex_mcmc(self, sentence):
        
        samples = [ "noun" ] * len(sentence)
        samples_list = []
        for i in range(0, 500):
            new_sample = list(samples)
            samples = self.generate_samples(new_sample, sentence)
            if(i > 200):
                samples_list.append(list(samples))
        final_samples =[]
        sample_arr = np.array(samples_list).transpose()
        for j in range(len(sentence)):
            s = max(Counter(sample_arr[j]).items(), key=lambda x: x[1])[0]
            final_samples.append(s)

        return final_samples

    '''
    Viterby uses bayes net where observed value word also depends on part of speech of prior word.
    For Viterbi algorithm, we have used result and backtrack as a matrix of rows equal to number of Part of speech available and columns as number of words in sentence
    For first word, we have calculated result value as initial probability * emission probability 
    For all later wods, we maximize over the probabilities calculated in previous word and its transition to current state
    
    
    '''
    def hmm_viterbi(self, sentence):
        pos = self.initial_prob.keys()
        result = []             # result to store final probability values
        backtrack = []          # STore track of POS
        for i in range(len(pos)):
            result.append([])
            backtrack.append([])
            for j in range(len(sentence)):
                result[i].append(0)
                backtrack[i].append(0)
        count = 0
        max_cost = 0
        list_costs = []
        for i, word in enumerate(sentence):
            for j, s in enumerate(pos):
                
                # When count is 0, that is for initial word we are just calculating initial probability * emission probability
                if(count == 0):
                    try:
                        out = log(self.initial_prob[s]) + log(self.emission_prob[s][word])
                        list_costs.append(out)
                        result[j][i] = out
                        backtrack[j][i] = [0]
                    except:
                        result[j][i] = log(0.0000001)
                        list_costs.append(result[j][i])
                        backtrack[j][i] = [0]
                else:
                    # Find the probability of last transaction and maximize over the transition * prior probabilities
                    list_costs = []
                    for k,s_trans in enumerate(pos):
                        try:
                            list_costs.append(result[k][i-1] + log(self.transition_prob[s_trans][s]))
                        except:
                            list_costs.append(result[k][i-1] + log(0.0000001))
                    max_cost = max(list_costs)

                    # Final probability is maximized value * emission probability
                    try:
                        result[j][i] = max_cost + log(self.emission_prob[s][word])
                    except:
                        result[j][i] = max_cost + log(0.0000001)
                    li = [x for x in backtrack[list_costs.index(max_cost)][i-1]]
                    li.append(list_costs.index(max_cost))
                    backtrack[j][i] = li
            count+=1
        # To find max cost from last word and append in the track
        list_costs = []
        for l in range(len(pos)):
            list_costs.append(result[l][-1])    
        max_cost = max(list_costs)
        path = [x for x in backtrack[list_costs.index(max_cost)][i]]
        path.append(list_costs.index(max_cost))
        path = path[1:] # Discard the first value as it is 0 for first word
        final_path =[]
        for i in range(0, len(path)):
            final_path.append(list(pos)[path[i]])   # Get actual list of POS
        return final_path
        
  


    # This solve() method is called by label.py, so you should keep the interface the
    #  same, but you can change the code itself. 
    # It should return a list of part-of-speech labelings of the sentence, one
    #  part of speech per word.
    #
    def solve(self, model, sentence):
        if model == "Simple":
            return self.simplified(sentence)
        elif model == "Complex":
            return self.complex_mcmc(sentence)
        elif model == "HMM":
            return self.hmm_viterbi(sentence)
        else:
            print("Unknown algo!")

