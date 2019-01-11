from nltk.tokenize import MWETokenizer, RegexpTokenizer
import numpy as np
import pandas as pd
import tensorflow as tf
"""
用邻近（模糊）算法去识别和剔除文本中的模板。
识别模板的能力一般，作为下一步deep learning模型的预处理
"""

def load_original_dataset(filename):
    sent_pairs = []
    with tf.gfile.GFile(filename, "r") as f:
        for line in f:
            ts = line.strip().split("\t")
            sent_pairs.append((ts[0], ts[1]))
    return pd.DataFrame(sent_pairs, columns=["sent_1", "sent_2"])


def write(filename, sent1,sent2,frq):
    with open(filename, 'w+') as f:
        temp = []
        for str1,str2,fre in zip(sent1,sent2,frq):
            if len(str1) > 1:
                str1 = " ".join(str1)
            if len(str2) > 1:
                str2 = " ".join(str2)
            temp.append(repr(str1))
            temp.append(repr(str2))
            temp.append(str(fre))
            print("\t".join(temp), file=f)
            temp.clear()
    f.close()


def mat(list1, list2):
    i = d = -1
    m = len(list1)
    n = len(list2)
    # construct the matrix, of all zeroes
    maxvalue = []
    currentMax = 3
    currentRow = currentCol = 0

    mat = [[0] * (n + 1) for row in range(m + 1)]
    # populate the matrix, iteratively
    for row in range(1, m + 1):
        for col in range(1, n + 1):
            temp = (2 if list1[row - 1] == list2[col - 1] else -1)
            mat[row][col] = max(0, mat[row][col - 1] + d, mat[row - 1][col - 1] \
                                + temp, mat[row - 1][col] + i)
            if currentMax < mat[row][col]:
                currentMax = mat[row][col]
                currentRow = row
                currentCol = col
    maxvalue=(currentMax,currentRow,currentCol)
    return mat, maxvalue


def backtrack(mat, list1, list2, row, col, list, templates):
    if row < 0 or col < 0:
        if len(list) != 0:
            templates.append(list)
        return templates
    if list1[row] == list2[col]:
        if list1[row - 1] == list2[col - 1]:
            list.append(list1[row])
            return backtrack(mat, list1, list2, row - 1, col - 1, list, templates)
        else:
            list.append(list1[row])
            temp_list = list.copy()
            templates.append(temp_list)
            list.clear()
            return backtrack(mat, list1, list2, row - 1, col - 1, list, templates)
    else:
        check = mat[row+1][col+1] +1
        if check == mat[row][col+1]:
            return backtrack(mat, list1, list2, row - 1, col, list, templates)
        elif check == mat[row+1][col]:

            return backtrack(mat, list1, list2, row, col-1, list, templates)
        elif check == mat[row][col]:

            return backtrack(mat, list1, list2, row - 1, col -1, list, templates)
        else:
            return templates

if __name__ == '__main__':
    str1 = "Negative neurologic review of systems, Historian denies confusion, dizziness, focal weakness, gait changes, headache."
    str2 = "Negative gastrointestinal review of systems, Historian denies abdominal pain, hematemesis, hematochezia, jaundice, melena, vomiting."
    tokenizer = RegexpTokenizer(r'\w+')
    sent1 = tokenizer.tokenize(str1)

    sent2 = tokenizer.tokenize(str2)
#     print(equal(sent1[0],sent2[3]))
    matrix, maxvalue = mat(sent1, sent2)
    for value in maxvalue:
         dist, row, col = value
         print(backtrack(mat, sent1, sent2, row - 1, col - 1, list, temp))
     print(current_row, sent1[current_row-1])
     print(current_col, sent2[current_col-1])

    # word1,word2 = sent1,sent2

    from nltk.tokenize import MWETokenizer

    tokenizer2 = MWETokenizer()

    data = load_original_dataset("../clinicalSTS.train.txt")
    str1_temp = []
    str2_temp = []
    freq_temp = []
    for i in range(len(data)):
        list = []
        temp = []
        freq = 0
        sentence1 = data["sent_1"][i]
        sentence2 = data["sent_2"][i]
        tokenizer = RegexpTokenizer(r'\w+')
        sent1 = tokenizer.tokenize(sentence1.lower())
        sent2 = tokenizer.tokenize(sentence2.lower())
        matrix, maxvalue = mat(sent1, sent2)
        template=[]

        dist, row, col = maxvalue
        word1, word2 = [],[]
        if dist > 1:
            for part in (backtrack(matrix, sent1, sent2, row - 1, col - 1, list, temp)):
                # print(part[::-1])
                freq += len(part)
                template.append([word for word in part])

        if not len(template)==0:
            haha=[]
            for item in template:
                haha+=item
            word1 = [wrd for wrd in sent1 if wrd not in haha]
            word2 = [wrd for wrd in sent2 if wrd not in haha]
            if len(word1)==0:
                word1.append('none')
            if len(word2)==0:
                word2.append('none')
            str1_temp.append(word1)
            str2_temp.append(word2)
            freq_temp.append(freq)

        else:
            str1_temp.append(sent1)
            str2_temp.append(sent2)
            freq_temp.append(0)

    JET_noT = pd.DataFrame()
    JET_noT['sent_1']=str1_temp
    JET_noT['sent_2']=str2_temp
    JET_noT['freq']=freq_temp
    JET_noT.to_csv('../no_temp_new.csv')
