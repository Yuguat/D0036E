def make_word_list():
    """Reads lines from a file and builds a list using append."""
    word_l = []
    dict = open('Wordlist.txt')
    for word in dict:
        word = word.strip()
        word_l.append(word)

        if len(word_l)==0:
            print("Word list is empty")
            return False

    return word_l
def revision(wordlist):

    if len(wordlist) != 0:
        try:
            word=wordlist[0]
            T.append(wordlist[wordlist.index(word[::-1])])
            Ta.append(word)
            wordlist.remove(word)
            wordlist.remove(wordlist[wordlist.index(word[::-1])])
            revision(wordlist)
        except:
            wordlist.remove(word)
            revision(wordlist)
            pass

    return T,Ta



T = []
Ta = []

list1=["atakan","yusuf","zeynepp","nakata"]
revision(list1)
print(T)
print(Ta)






