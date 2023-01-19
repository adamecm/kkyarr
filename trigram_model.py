import math
import time
start = time.time()
text_path = "./train.txt"
dict_path = "./cestina"
export_path = "./lm.arpa"

LEGAL_WORDS = set()
with open(dict_path,"r",encoding="windows-1250") as f:
  for line in f.readlines():
    LEGAL_WORDS.add(line.strip("\n"))
LEGAL_WORDS.add("</s>")
#LEGAL_WORDS.add("<s>")
# lap1 = time.time()
# print("l1",lap1-start)

word_count = 0
sentence_count = 0
unigram_count = dict()
bigram_count = dict()
trigram_count = dict()
with open(text_path,"r",encoding="windows-1250") as f:
  for line in f.readlines():
    sentence = ("<s> " + line.strip("\n") + " </s>").split()
    
    sentence_count += 1

    for idx in range(1,len(sentence)):
      if sentence[idx] not in LEGAL_WORDS:
        sentence[idx] = "<unk>"
      
      word_count += 1

      # UNIGRAMS
      word = sentence[idx]
      unigram_count[word] = unigram_count.get(word,0) + 1  
      
      #BIGRAMS
      if idx > 0:
        bigram = (sentence[idx-1],sentence[idx])
        bigram_count[bigram] = bigram_count.get(bigram,0) + 1  

      #TRIGRAMS
      if idx > 1:
        trigram = (sentence[idx-2],sentence[idx-1],sentence[idx])
        trigram_count[trigram] = trigram_count.get(trigram,0) + 1  

#PRUNE TRIGRAMS
trigram_count = {key:value for key,value in trigram_count.items() if value > 1}

#ESTIMATE LIKELIHOODS    
#UNIGRAMS
print("########################UNIGRAMY########################")
unigram_ML = unigram_count.copy()
for word,count in unigram_count.items():
  unigram_ML[word] = math.log10(count/word_count)

print("uni a :",unigram_ML["a"])
print("uni <unk> :", unigram_ML["<unk>"])
print("uni </s> :",unigram_ML["</s>"])
print()
print("########################BIGRAMY########################")
#BIGRAMS
bigram_ML = bigram_count.copy()
for bigram in bigram_count.keys():
  word = bigram[0]
  if word == "<s>":
    denom = sentence_count
  else:
    denom = unigram_count[word]
  bigram_ML[bigram] = math.log10(bigram_count[bigram]/denom)

print("bi <s> <unk> :", bigram_ML[("<s>", "<unk>")])
print("bi a o :", bigram_ML[("a", "o")])
print()
print("########################TRIGRAMY########################")

#TRIGRAMS
trigram_ML = trigram_count.copy()
for trigram in trigram_count.keys():
  wordset = tuple(trigram[0:2])
  denom = bigram_count[wordset]
  trigram_ML[trigram] = math.log10(trigram_count[trigram]/denom)

print("tri <s> <unk> <unk> :", trigram_ML[("<s>", "<unk>", "<unk>")])
print("tri a tak se :", trigram_ML[("a", "tak", "se")])
print("tri a za této :", trigram_ML[("a", "za", "této")])
# print("bi a o :", trigram_ML["a o"])
# lap3 = time.time()
# print("l3",lap3-lap1)
# print("full",time.time()-start)

with open(export_path,"w") as f:
  f.write("\\data\\\n")
  f.write(f"ngram 1={len(unigram_count)}\n")
  f.write(f"ngram 2={len(bigram_count)}\n")
  f.write(f"ngram 3={len(trigram_count)}\n")
  f.write("\n")
  f.write("\\1-grams:\n")
  for key, val in unigram_ML.items():
    f.write(str(val)+ " " + key + "\n")
  f.write("\n")
  f.write("\\2-grams:\n")
  for key, val in bigram_ML.items():
    f.write(str(val)+ " " + key[0] + " " + key[1] + "\n")
  f.write("\n")
  f.write("\\3-grams:\n")
  for key, val in trigram_ML.items():
    f.write(str(val) + " " + key[0] + " " + key[1]+ " " + key[2] + "\n")
  f.write("\n\\end\\")
    
  