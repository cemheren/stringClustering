import pickle
import re
import string

from math import log

# Build a cost dictionary, assuming Zipf's law and cost = -math.log(probability).
words = open("words_by_frequency.txt").read().split()
# words = pickle.load(open("words", 'rb'))
wordcost = dict((k, log((i+1)*log(len(words)))) for i, k in enumerate(words))
maxword = max(len(x) for x in words)

def camel_case_split(identifier):
    matches = re.finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', identifier)
    return [m.group(0) for m in matches]

def infer_spaces(s):
    """Uses dynamic programming to infer the location of spaces in a string
    without spaces."""

    # Find the best match for the i first characters, assuming cost has
    # been built for the i-1 first characters.
    # Returns a pair (match_cost, match_length).
    def best_match(i):
        candidates = enumerate(reversed(cost[max(0, i-maxword):i]))
        return min((c + wordcost.get(s[i-k-1:i], 9e999), k+1) for k,c in candidates)

    # Build the cost array.
    cost = [0]
    for i in range(1,len(s)+1):
        c, k = best_match(i)
        cost.append(c)

    # Backtrack to recover the minimal-cost string.
    out = []
    i = len(s)
    while i>0:
        c,k = best_match(i)
        assert c == cost[i]
        out.append(str(s[i-k:i]))
        i -= k

    return reversed(out)


def sentence_to_word_array(sentence="", remove_punctuation=True, to_lowercase=True, should_infer_spaces=True, should_camel_case_split=True):
    return_array = list()
    split = list()

    if remove_punctuation:
        split = filter(None, re.split("[,\W\x03\x07\x0b\n\t\r.()' \-!?:/#@*^%&$<>;\"`~{}-]+", sentence))

        for word in split:

            if should_camel_case_split:
                inferred = camel_case_split(word.lower())
                for i in inferred:
                    if len(i) > 2 and not i.lower() in return_array:
                        return_array.append(i.lower())

            if remove_punctuation:
                word = word.translate(string.punctuation)

            if should_infer_spaces:
                inferred = infer_spaces(word.lower())
                for i in inferred:
                    if len(i) > 2 and not i.lower() in return_array:
                        return_array.append(i.lower())

            if len(word) > 0 and not i.lower() in return_array:
                return_array.append(word.lower())

    return return_array


def words_to_numbers(input_matrix=[[]], file_to_write=""):

    return_matrix = list()

    words = dict()
    current_count = 1  # start from 1

    for i in range(len(input_matrix)):
        row = input_matrix[i]

        return_matrix.append(list())

        for k in range(len(row)):
            col = row[k]

            if col not in words:
                words[col] = current_count
                current_count += 1

            return_matrix[i].append(words[col])

    if file_to_write != "":
        pickle.dump(words, open(file_to_write, 'wb'))

    return return_matrix, words


def words_to_numbers_from_old_words_dict(input_matrix=[[]], words=dict(), unk_integer=-1, file_to_read=""):

    if file_to_read != "":
        words = pickle.load(open(file_to_read, 'rb'))

    return_matrix = list()

    for i in range(len(input_matrix)):
        row = input_matrix[i]

        return_matrix.append(list())

        for k in range(len(row)):
            col = row[k]

            value = unk_integer
            if col in words:
                value = words[col]

            return_matrix[i].append(value)

    return return_matrix
