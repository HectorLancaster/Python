import string
alphabet = string.ascii_letters

count_letters = dict()
sentence = "Jim quickly realized that the beautiful gowns are expensive"
value = 0
for letters in alphabet:
    if letters in sentence:
        for i in range(len(sentence)):
            if letters is sentence[i]:
                value += 1
            count_letters[letters] = value
        value = 0
    else:
        count_letters[letters] = 0
for letters in count_letters.keys():
    print(letters, count_letters[letters])