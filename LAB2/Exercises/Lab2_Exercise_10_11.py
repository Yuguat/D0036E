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



def in_bisect(word_list, word):
    if len(word_list) == 0:
        return False

    i = len(word_list) // 2
    if word_list[i] == word:

        return True

    if word_list[i] > word:

        return in_bisect(word_list[:i], word)
    else:

        return in_bisect(word_list[i + 1:], word)




def reverse_pair(Wordlist):
    if len(Wordlist) == 0:
        return False
    word = Wordlist[0]
    rev_word = word[::-1]
    if in_bisect(Wordlist, rev_word) and (word!=rev_word):

        Wordlist.remove(word)
        Wordlist.remove(rev_word)
        print(word, rev_word)

    else:
        Wordlist.remove(word)

    reverse_pair(Wordlist)

Wordlist=make_word_list()
#reverse_pair(Wordlist)
for word in Wordlist:
    rev_word = word[::-1]
    if in_bisect(Wordlist, rev_word):
        print(word ,rev_word)







