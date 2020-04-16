def counter(sentence):
    """

    Parameters
    ----------
    *text : input a string " "

    Returns
    -------
    An alphabetically ordered list of letters used, both upper and lower case,
    in the user's input text

    """
    import string
    alphabet = string.ascii_letters
    value = 0
    count_letters = dict()
    #sentence = str(text)
    for letters in alphabet:
        if letters in sentence:
            for i in range(len(sentence)):
                if letters is sentence[i]:
                    value += 1
                    count_letters[letters] = value
            value = 0
    for letters in count_letters.keys():
       print(letters, count_letters[letters])
       
counter("hello I think that this is a test")