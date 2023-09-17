import string as s
P=tuple(s.punctuation)


dict = open('13_1_text.txt')
dict1=[]
for word in dict:
    word2=word.split()

    for i_w in word2:
        for i_s in i_w:
            if i_s in P:
               i_w=i_w.replace(i_s, "")
        dict1.append(i_w.lower())




print(dict1)