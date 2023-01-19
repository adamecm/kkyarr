import copy
import numpy as np
import re

LEAVES_PATH = "./leaves.txt"
VOCAB_PATH = "./vocab"
ARPA_PATH = "./lm_arpa.txt"
OBSERVATIONS_PATH = "./00170005_14.txt"

LEAVES_ENCODING = "windows-1250"
VOCAB_ENCODING = "windows-1250"
ARPA_ENCODING = "utf-8"
OBSERVATIONS_ENCODING = "windows-1250"
 
BETA = 5
LGAMMA = -np.log10(0.01)


#load phoneme data
char_idx = dict() #phonemes in word net are represented by index 
trans_probs = list() #loop and trans probabilities
with open(LEAVES_PATH,"r",encoding=LEAVES_ENCODING) as file:
  for idx,line in enumerate(file.readlines()):
    splitline = line.split()
    char_idx[splitline[0]] = idx
    trans_probs.append((-np.log10(float(splitline[1])),-np.log10(float(splitline[2]))))

word_net = list() #list of lists of phonemes, represented as numbers
wordlist = list() #list of all words from vocab

#load vocabulary
with open(VOCAB_PATH,"r",encoding=VOCAB_ENCODING) as file:
  for line in file.readlines():
    temp = line.split("		")
    wordlist.append(temp[0])
    phonemes = temp[1]
    word_net.append([char_idx[x] for x in phonemes.split()])
  word_net.append([char_idx["#"]])
wordlist.append("#")#silence is not in vocab, has to be added

observations_B = list() #observation vectors B
#load observation vectors
with open(OBSERVATIONS_PATH,"r",encoding=OBSERVATIONS_ENCODING) as file:
  for line in file.readlines():
    observations_B.append([-np.log10(float(x)) for x in line.split()])


#load language model in an arpa format using regex
language_model = dict() #dictionary of dictionaries, contains uni- bi- and trigrams
with open(ARPA_PATH,"r",encoding=ARPA_ENCODING) as file:
  lm = "".join(file.readlines())
  all_unigrams = re.findall("-?\d+.\d+ \S+\n",lm)
  chunk_dict = dict()
  for element in all_unigrams:
    tmp = element.strip("\n").split()
    chunk_dict[tmp[1]] = float(tmp[0])
  language_model["unigrams"] = chunk_dict

  all_bigrams = re.findall("-?\d+.\d+ \S+ \S+\n",lm)
  chunk_dict = dict()
  for element in all_bigrams:
    tmp = element.strip("\n").split()
    chunk_dict[(tmp[1],tmp[2])] = float(tmp[0])
  language_model["bigrams"] = chunk_dict
  

  all_trigrams = re.findall("-?\d+.\d+ \S+ \S+ \S+\n",lm)
  chunk_dict = dict()
  for element in all_trigrams:
    tmp = element.strip("\n").split()
    chunk_dict[(tmp[1],tmp[2],tmp[3])] = float(tmp[0])
  language_model["trigrams"] = chunk_dict

unigrams = language_model["unigrams"] #we only use unigrams in our task

#initialize phi net
phi_net = copy.deepcopy(word_net) # phi net has the same structure as word net
token_net = copy.deepcopy(word_net)# token net has the same structure as word net
for w in range(len(phi_net)):
  phi_net[w][0] = observations_B[0][word_net[w][0]] #first row (time 0), token number 0
  token_net[w][0] = 0
  for phoneme_idx in range(1,len(phi_net[w])):
    phi_net[w][phoneme_idx] = np.infty
    token_net[w][phoneme_idx] = 0

tokens = list()
for t in range(1,len(observations_B)): #t = 2 to T (392), every observation vector
  ends = list()
  for idx in range(len(word_net)):
    phi = phi_net[idx][-1] # cost of words end (phi)
    phoneme = word_net[idx][-1] # actual phoneme at the words end
    penalisation = LGAMMA - BETA*unigrams.get(wordlist[idx],0) # we add this to penalize adding a word
    ends.append(phi + trans_probs[phoneme][1] + penalisation)
  
  min_cost_val = min(ends)
  min_cost_idx = ends.index(min_cost_val)
  tokens.append((wordlist[min_cost_idx],token_net[min_cost_idx][-1]))
  min_cost_token = len(tokens) - 1
  
  for w in range(len(phi_net)): #every word
    old_phi = phi_net[w].copy()
    old_token = token_net[w].copy()
    word = word_net[w]
    cost = min(min_cost_val,(old_phi[0] + trans_probs[word[0]][0]))
    token = min_cost_token if (cost == (min_cost_val)) else old_token[0]
    phi_net[w][0] = cost + observations_B[t][word[0]]# + penalisation
    token_net[w][0] = token
    for j in range(1,len(word)):#second to last phoneme of each word
      prev_phi = old_phi[j-1] + trans_probs[word[j-1]][1] #prev + trans
      prev_token = old_token[j-1]
      current_phi = old_phi[j] + trans_probs[word[j]][0] #same + loop
      current_token = old_token[j]
      cost = min(prev_phi,current_phi)
      token = prev_token if (cost == prev_phi) else current_token
      phi_net[w][j] = cost + observations_B[t][word[j]]
      token_net[w][j] = token

ends = list()
for idx in range(len(word_net)):
  phi = phi_net[idx][-1] # cost of words end
  phoneme = word_net[idx][-1] # actual phoneme at the words end
  word = wordlist[idx] #word from the language model
  penalisation = LGAMMA - BETA*unigrams.get(word,0) # we add this to penalize adding a word
  ends.append(phi + trans_probs[phoneme][1] + penalisation)
final_min_val = min(ends)
final_min_idx = ends.index(final_min_val)
final_min_token = token_net[final_min_idx][-1]
print("Minimal cost = ",final_min_val)

#print words by backtracking tokens
previous_token_index = final_min_token
spoken_words = [wordlist[final_min_idx]]
while(previous_token_index != 0):
    spoken_words.append(tokens[previous_token_index][0])
    previous_token_index = tokens[previous_token_index][1]
spoken_words.reverse()
print("Result:")
print(spoken_words)

#chceme: # šéf roste ovšem šest požadavek této situace není devalvaci důvod #