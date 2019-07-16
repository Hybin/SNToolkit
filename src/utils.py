def annotate(word):
    tokens = list()

    if len(word) == 1:
        tokens.append([word, "S"])
    else:
        for token in word:
            if word.index(token) == 0:
                tokens.append([token, "B"])
            elif word.index(token) == len(word) - 1:
                tokens.append([token, "E"])
            else:
                tokens.append([token, "M"])

    return tokens


def get_length(array):
    lst = [item for item in array]
    sentence = []

    for item in lst:
        if item == 0:
            continue
        else:
            sentence = lst[lst.index(item):]
            break

    return len(sentence)
