# define functions, using tuples to return multiple values
def intersect(s1, s2):
    """

    Parameters
    ----------
    s1 : list one
    s2 : list two

    Returns
    -------
    res : list of common values between s1 and s2

    """
    res = []
    for x in s1:
        if x in s2:
            res.append(x)
    return res

def password(length):
    """

    Parameters
    ----------
    length : number of characters in password

    Returns
    -------
    pw : random password of given length

    """
    import random
    pw = str()
    characters = "abcdefghijklmnopqrstuvwxyz" + "123456789"
    for i in range(length):
        pw = pw + random.choice(characters)
    return pw
