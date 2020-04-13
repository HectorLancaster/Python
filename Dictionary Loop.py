# Dictionary Loop
# Define dictionary
album_ratings = {"Back in Black": 8.5, "Riot!": 7.5}

# For each key in the dictionary, if it's value is greater than 8 then print text A, esle print text B
for key in album_ratings:
    if album_ratings[key] > 8:
        print(key, "is an amazing album!")
    else:
        print(key, "is not such an amazing album...")