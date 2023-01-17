# checks if list already contains the word pair
def is_pair_in_list(all_pairs, pair):
    for p in all_pairs:
        if (p[0] == pair[0]) and p[1] == pair[1]:
            return True
    return False


# returns word list of noun pairs of Zhao et al. and 100 self-created name pairs
def get_gender_word_list():
    word_list = []
    # https://github.com/uclanlp/corefBias/blob/master/WinoBias/wino/generalized_swaps.txt
    file_wordlist = open('./data_prep/data/wordpairs/cda_word_pairs_gender.txt', 'r', encoding="utf-8") 
    
    lines_wordlist = file_wordlist.readlines()
    for line in lines_wordlist:
        word_pair = line.split()
        #print(word_pair)
        word_list.append(word_pair[0])
        word_list.append(word_pair[1])

    # https://github.com/uclanlp/corefBias/blob/master/WinoBias/wino/extra_gendered_words.txt
    file_wordlist = open('./data_prep/data/wordpairs/cda_word_pairs_gender_extra.txt', 'r', encoding="utf-8") 
    
    lines_wordlist = file_wordlist.readlines()
    for line in lines_wordlist:
        word_pair = line.split()
        if not is_pair_in_list(word_list, word_pair):
            word_list.append(word_pair[0])
            word_list.append(word_pair[1])
            #word_list.append([word_pair[1], word_pair[0]]) # both 'dircetions' needed: (male, female) and (female, male)
        
    # https://www.ssa.gov/oact/babynames/limits.html
    file_wordlist = open('./data_prep/data/wordpairs/cda_word_pairs_names.txt', 'r', encoding="utf-8") 
    
    lines_wordlist = file_wordlist.readlines()
    for line in lines_wordlist:
        word_pair = line.split()
        if not is_pair_in_list(word_list, word_pair):
            word_list.append(word_pair[0])
            word_list.append(word_pair[1])
    word_list.append("his")
    word_list.append("her")
    word_list.append("seth")
    word_list.append("sarah")
    word_list.append("himself")
    word_list.append("herself")
    word_list.append("male")
    word_list.append("female")
    
    return word_list

def get_gender_word_pairs():
        word_pairs = []
        # https://github.com/uclanlp/corefBias/blob/master/WinoBias/wino/generalized_swaps.txt
        # creates list with word pairs --> [ [pair1[0], pair1[1]] , [pair2[0], pair2[1]] , ... ]
        file_wordlist = open('./data_prep/data/wordpairs/cda_word_pairs_gender.txt', 'r', encoding="utf-8") 
        
        lines_wordlist = file_wordlist.readlines()
        for line in lines_wordlist:
            word_pair = line.split()
            #print(word_pair)
            word_pairs.append(word_pair)

        # https://github.com/uclanlp/corefBias/blob/master/WinoBias/wino/extra_gendered_words.txt
        # appends additional word pairs from extra file
        file_wordlist = open('./data_prep/data/wordpairs/cda_word_pairs_gender_extra.txt', 'r', encoding="utf-8") 
        
        lines_wordlist = file_wordlist.readlines()
        for line in lines_wordlist:
            word_pair = line.split()
            if not is_pair_in_list(word_pairs, word_pair):
                word_pairs.append(word_pair)
                word_pairs.append([word_pair[1], word_pair[0]]) # both 'dircetions' needed: (male, female) and (female, male)
            
        # https://www.ssa.gov/oact/babynames/limits.html
        # gets the top 100 names of 2019 for boys and girls and appends the pairs (male, female) and (female, male) to the word pair list
        file_wordlist = open('./data_prep/data/wordpairs/cda_word_pairs_names.txt', 'r', encoding="utf-8") 
        
        lines_wordlist = file_wordlist.readlines()
        for line in lines_wordlist:
            word_pair = line.split()
            if not is_pair_in_list(word_pairs, word_pair):
                word_pairs.append(word_pair)
        
        # do some adjustments
        word_pairs.append(["his", "her"])
        word_pairs.append(["her", "his"])
        word_pairs.append(["seth", "sarah"])
        word_pairs.append(["sarah", "seth"])
        word_pairs.append(["himself", "herself"])
        word_pairs.append(["herself", "himself"])
        word_pairs.append(["male", "female"])
        word_pairs.append(["female", "male"])
        return word_pairs
