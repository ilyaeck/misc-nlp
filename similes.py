
# coding: utf-8

# **the algorithm:**
# #take a big corpus with trite similes (The Daily Mail, http://cs.nyu.edu/~kcho/DMQA/, or a pulp fiction book), prepare clean sentences, traverce throug the corpus sen by sen, POS-tagging each and looking for sentences containing prepositions "like" and "as" (tag = "IN") and exclude "as soon as, as well as, as usual, such as, as of yet, as much, like that, like this.." ((alternatively, use dependency parser to accurately cut out a phrase. but it's a pain and may be an overkill)) Add these sentences to a target corpus. Cut out a simile candidate out of each sentence; optionally: replace "likes" and "ases" with a "comparator". 
# #approach 1:
# #Use fuzzywazzy to do fuzzy matching of the simile candidates across the corpus. Find those that at least 98% similar and appear multiple times (over 10) across the corpus - those are thrite similes (or common grammatical constuctions containing 'like' or 'as' that we missed during cleaning). Build a corpus of trite similes. With a testing set, repeat all steps up to fuzzywazzy. Then, instead of fuzzy matching candidates across the testinf set, fuzzy match them with the trite similes corpus. Highlight (tag) if a match is found.  
# #approach 2: 
# #Sort words in each set alphabetically. Then build an n-gram counter (may be plot a histogram) - a dictionary with an n-gram as a key and how many times is appears in the corpus as a value. In a new text, repeat all steps up to the last one and then find new n-grams in the dictionary. If the new n-grams are among the most frequently met n-grams in the corpus, these n-grams constitute trite similes. Then use them as a trite similes corpus and compare to the testing set as described in the approach 1. 
# 

# In[1]:

import nltk

min_simile_freq = 5
train_dir_name = '../raw_data/similes_train/' 
test_dir_name = '../raw_data/similes_test/' 


# from nltk.parse.stanford import StanfordDependencyParser
# path_to_jar = '/Development/Projects/Magnifis/3rd_party/NLU/stanford-corenlp-full-2013/stanford-corenlp-3.2.0.jar'
# path_to_models_jar ='/Development/Projects/Magnifis/3rd_party/NLU/stanford-corenlp-full-2013/stanford-corenlp-3.2.0-models.jar'
# dependency_parser = StanfordDependencyParser(path_to_jar=path_to_jar, path_to_models_jar=path_to_models_jar)

# result = dependency_parser.raw_parse('I shot an elephant in my sleep')
# dep = result.next()
# list(dep.triples())


# In[2]:

import os
import io
import codecs
import re


regex_filter = r"(as soon)|(as well)|(as if)|(as \w+ as possible)|(as before)|(as long)|(as usual)|(as ever)|(as a result)|(such as)|(as of yet)|(as much)|(as many)|(like that)|(like this)|(like you)|(like me)|(like him)|(like us)|(like her)|(like everything else)|(like everyone else)|(anybody like)|(anyone like)"



# In[3]:

from tqdm import tqdm

def get_raw_text_data(input_dir):  
    fList=os.listdir(input_dir)
    # Create a list of file names with full path, eliminating subdirectory entries
    fList1 = [os.path.join(input_dir, f) for f in fList if os.path.isfile(os.path.join(input_dir, f))] 
    
    #max_files = 1000 #remove to get the entire corpus
    raw_corpus = ''
    for file in tqdm(fList1): #[0:max_files] 
        with codecs.open(file, 'r', 'latin_1') as f: 
                                        # 'utf-8') as f:
        #with open(file, encoding="utf8") as f:
            raw_corpus += ''.join(f.read())  
    corpus = re.sub(r"(\n|\r)+""|(@\w+)+", ' ', raw_corpus) #remove backslashes and words starting with @
    #corpus = re.sub(r"(as soon)+" "|(as well)+" "|(as if)+" "|(as quickly as possible)+" "|(as long)" "|(as usual)+" "|(such as)+" "|(as of yet)+" "|(as much)+" "|(as many)+" "|(like that)+" "|(like this)+" "|(like you)+" "|(like me)+" "|(like him)+" "|(like us)+" "|(like her)+" "|(anybody like)+" "|(anyone like)+", "", corpus)
    return corpus


# In[4]:

def tokenize_text(corpus, regex_filter, do_tokenize_words=True):
    sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    raw_sents = sent_tokenizer.tokenize(corpus) # Split text into sentences 
    if regex_filter:
        raw_sents = [sent for sent in raw_sents if not re.search(regex_filter, sent)]
    if do_tokenize_words:
        result = [nltk.word_tokenize(sent) for sent in raw_sents]
    else: 
        result = raw_sents
    return result


# In[5]:

def extract_simile_candidates(sentences):
    comparisons = []
    for sent in sentences:
        if not 'like' in sent and not 'as' in sent: 
            continue 
        # exlude a single 'as', leaving in only '...as ... as...'
        if not 'like' in sent and len([word for word in sent if word=='as']) == 1: 
            continue
        pos_tagged = nltk.pos_tag(sent)
        for pair in pos_tagged:
            if pair[1] == 'IN' and (pair[0] == 'like' or pair[0] == 'as'):
                comparisons.append(pos_tagged)
    return comparisons


# In[6]:

def filter_candidates(all_candidates):
    similes_candidates = []
    punkt = set(['.',',','-',':',';','!','?', '"', '\'', ')', '(', '%', '#', '[', ']', '@'])
    key_pos_tags = set(['NN', 'NNS', 'NNP']) #, 'VB', 'VBN', 'VBD', 'VBG']) # noun or verb
    for tagged_sent in all_candidates:
        start_index = -1
        words_after = -1
        sent = [pair[0] for pair in tagged_sent]
        pos_tags = [pair[1] for pair in tagged_sent]
        if 'like' in sent:
            start_index = sent.index('like')
            #two_words_before_like = max(0, index_of_like - 4)
            words_after = min(len(sent), start_index + 6)
        elif 'as' in sent:
            start_index = sent.index('as')
            words_after = min(len(sent), start_index + 8)

        if start_index >= 0 and words_after > 0:
            index_of_punkt = 0
            for i in range(start_index, words_after): 
                if sent[i] in punkt: 
                    index_of_punkt = i
                    break 

            if index_of_punkt > start_index: 
                words_after = min(words_after, index_of_punkt)
            if not(not key_pos_tags.intersection(set(pos_tags[start_index:words_after]))): # make sure at least one key pos tag is present
                similes_candidates.append(sent[start_index:words_after])
    return similes_candidates


# In[7]:

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
stop_words = set(['a', 'an', 'and', 'or', 'the',                   'his', 'her', 'my', 'your', 'their', 'our',                   'i', 'you', 'he', 'she', 'it', 'they', 'who', 'that', 'whose',                   'is', 'are', 'was', 'will', 'would',                   '.',',','-',':',';','!','?', '"', '\'', ')', '(', '%', '#', '[', ']', '@'])

def preprocess_words(wordlist): 
    wordset = set([])
    for word in wordlist: 
        word = word.lower()
        if word not in stop_words and len(word) > 1: 
            if word != 'as':
                word = lemmatizer.lemmatize(word)
            if word == 'like' or word == 'as': 
                word = '$cmpr'
            wordset.add(word)
    return wordset 
        
''' Precomputes a corpus (phrase search index) for a given list of phrases
    Optimization: create a data structure to speed up fuzzy matching as follows: 
    {'word' : [i, j, k, ...]}, where i, j, k are the row indices of all phrases containing 'word'. 
    For each new search query, we prefetch the relevant rows based in the words in that query, 
    prior to fuzzy matching. 
'''
def init_corpus_2match(wordlists): 
    lookup = {}
    all_wordsets = []
    for words in wordlists: # for each phrase (word list)
        if not words:
            continue
        wordset = preprocess_words(words)
        if not(not wordset):
            i_row = len(all_wordsets)   
            all_wordsets.append(wordset)
            
            # update loookup index (dictionary of word to corpus row id)
            for word in wordset: 
                if word not in lookup: 
                    lookup[word] = [i_row]
                else: 
                    lookup[word].append(i_row)
    return (all_wordsets, lookup) 


''' Returns a list of matches for 'phrase' in 'wordsets' with 'min_similarity' 
'''
def fuzzy_match(words_in, search_index, min_similarity): 
    # init 
    phraset = preprocess_words(words_in)
    relevant_corpus_rows = search_index
    
    # prepare relevant subset of search index
    # the data could be in 2 different representations
    if isinstance(search_index, tuple): 
        corpus = search_index[0]
        lookup = search_index[1]

        # prefetch relevant corpus rows 
        relevant_corpus_row_ids = set([])
        for word in phraset: 
            if word not in lookup:
                continue
            row_ids = lookup[word]
            for i in row_ids:
                relevant_corpus_row_ids.add(i)    
        relevant_corpus_rows = [corpus[i] for i in relevant_corpus_row_ids]  
        
    # actually search
    nb_input = len(phraset)
    matches = []
    for wordset in relevant_corpus_rows: 
        intersect = phraset.intersection(wordset)
        n = len(intersect)
        if n/min(nb_input, len(wordset)) >= min_similarity and not(n < 2 and next(iter(intersect))=='$cmpr'): 
            #print(wordset)
            matches.append(wordset)
    return matches


# In[8]:

import operator

def train_similes_corpus(candidates):
    corpus_2match = init_corpus_2match(candidates)
    covered = set([])
    count_dict = {}
    for cand in candidates:
        if not cand: 
            continue
        phrase = ' '.join(cand)
        if phrase in covered:
            continue
        covered.add(phrase)
        result = fuzzy_match(cand, corpus_2match, 0.75)
        #print("result is {}".format(result))
        if result:
            count_dict[phrase] = len(result)
    
    sorted_counts = sorted(count_dict.items(), key=operator.itemgetter(1))
    sorted_counts.reverse()
    return count_dict, sorted_counts


# In[9]:

from tqdm import tqdm

def aggregate_similes_candidates(input_dir):  
    fList=os.listdir(input_dir)
    # Create a list of file names with full path, eliminating subdirectory entries
    fList1 = [os.path.join(input_dir, f) for f in fList if os.path.isfile(os.path.join(input_dir, f))] 
    
    #max_files = 1000 #remove to get the entire corpus
    all_candidates = []
    for i in tqdm(range(len(fList1))): #[0:max_files] 
        file = fList1[i]
        with codecs.open(file, 'r', 'latin_1') as f: 
                                        # 'utf-8') as f:
        #with open(file, encoding="utf8") as f:
            raw_text = ''.join(f.read()) 
            text = re.sub(r"(\n|\r)+""|(@\w+)+", ' ', raw_text) #remove backslashes and words starting with @
            sentences = tokenize_text(text, regex_filter)
            similes_candidates = extract_simile_candidates(sentences)
            similes_candidates = filter_candidates(similes_candidates)
            all_candidates.extend(similes_candidates)
    return all_candidates


# ## Extract simile candidates from raw text  

# In[10]:

from sklearn.externals import joblib

def train(input_dir, min_simile_freq): 
    similes_candidates = aggregate_similes_candidates(input_dir)
    count_dict, sorted_counts = train_similes_corpus(similes_candidates)

    # create actual corpus and save 
    top_similes_corpus = init_corpus_2match([item[0].split(' ') for item in count_dict.items() if item[1] >= min_simile_freq])
    # save 
    joblib.dump(top_similes_corpus, "top_similes_corpus.v2.pkl")
    return similes_candidates, sorted_counts


# ## Train 

# In[ ]:


similes_candidates, sorted_counts = train(train_dir_name, min_simile_freq)
sorted_counts


# In[ ]:

similes_candidates[0:5]


# In[ ]:

sorted_counts


# ## Test 

# In[31]:

def extract_tagged_simile_sents(sentences):
    simile_sents = []
    for sent in sentences: 
        if not re.search("<rule1s>", sent):
            continue
        sent = re.sub(r"(<rule1s>)|(</rule1s>)", "", sent)
        simile_sents.append(nltk.pos_tag(nltk.word_tokenize(sent))) 
    return simile_sents


# Test last step: (pseudo-)"classification" of simile_candidates
def test(data_dir, similes_corpus, min_simile_freq): 
    raw_corpus = get_raw_text_data(data_dir)
    sentences = tokenize_text(raw_corpus, None, do_tokenize_words=False)
    true_simile_sents = extract_tagged_simile_sents(sentences)
    
    nb_true_pos = 0
    false_pos = []
    false_neg = []
    for true_simile_sent in true_simile_sents:
        # sent_words = [pair[0] for pair in tagged_sent]
        predicted_simile = False
        sent = [pair[0] for pair in true_simile_sent] # remove POS tags 
        nb_matches = 0
        nb_true_pos += 1
        simile_candidates = filter_candidates([true_simile_sent])
        if not (not simile_candidates): 
            simile_candidate = simile_candidates[0]
            matches = fuzzy_match(simile_candidate, similes_corpus, 0.75)
            nb_matches = len(matches)
        
        if nb_matches >= min_simile_freq:
            predicted_simile = True

        if not predicted_simile:
            false_neg.append(' '.join(sent))
#         else 
#             print("'{}' is NOT a trite simile".format(cand))
    precision = nb_true_pos / (nb_true_pos + len(false_pos))
    recall = nb_true_pos / (nb_true_pos + len(false_neg))
    print("=== Claddification Report ===")
    print("Precision = {}".format(precision))
    print("Recall = {}".format(recall))
    print("=============================")
    print ("-- False Negatives --")
    for neg in false_neg:
        print(neg)
        


# In[32]:

similes_corpus = joblib.load("top_similes_corpus.v2.pkl")
test(test_dir_name, similes_corpus, 1)


# In[16]:

# Misc unit tests 
raw_corpus = get_raw_text_data(test_dir_name)
sentences = tokenize_text(raw_corpus, None, do_tokenize_words=False)
true_simile_sents = extract_tagged_simile_sents(sentences)
simile_candidates = filter_candidates(true_simile_sents)


# In[17]:

simile_candidates[0:2]


# ## Backup code 

# In[18]:

# import fuzzywuzzy
# from fuzzywuzzy import fuzz
# from fuzzywuzzy import process


# In[19]:

# choices = []
# for each in similes_candidates:
#     choices.append(" ".join(each))


# In[20]:

# count_dict = {}

# for string in set(choices):
#     result = process.extract(string, choices, limit=1000) #default limit = 5
#     num_matches = 0
#     for each in result:
#         if each[1] > 98:
#             num_matches +=1
#     count_dict[string] = num_matches


# In[21]:

# write 
# from sklearn.externals import joblib
# joblib.dump(count_dict, "count_dict_output.pkl")


# In[22]:

# count_dict = sorted(count_dict.items(), key=operator.itemgetter(1))
# count_dict.reverse()


# In[23]:

#read 
#count_dict_fromfile = joblib.load("count_dict_output.pkl")

