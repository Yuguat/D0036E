def sed():
    try:
        pattern = input("Please your pattern")
        replace = input("Please your replace")
        source = '14_1_Test.txt'
        dest = source + '_Corrected'
        fin = open(source, 'r')
        fout = open(dest, 'w')

        for line in fin:
            line = line.replace(pattern, replace)
            fout.write(line)

        fin.close()
        fout.close()
    except:
        print("Please check your txt file")

sed()