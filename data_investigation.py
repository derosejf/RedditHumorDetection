import csv
import numpy as np
import os
import pandas as pd
import re

data_dir = 'data/'

missing = ['[removed]', '[deleted]']

## Remove 'edit:' or 'edit1:'
## strip jokes in the dataset class
# Weird unicode characters?

edits = ['edited to say', 'edit:', 'edit1:', 'credit:']

for name in ['train.tsv', 'dev.tsv', 'test.tsv']:
    jokes = []
    labels = []

    with open(os.path.join(data_dir, name), 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=',')

        for (i, line) in enumerate(reader):
            text = line[3]
            label = line[1]
            jokes.append(text)
            labels.append(label)

    good_jokes = []
    good_labels = []
    pos = 0
    neg = 0
    for ix, joke in enumerate(jokes):
        joke = joke.lower()

        # remove missing jokes
        if any(bad in joke for bad in missing):
            continue

        # split, strip and recombine
        split = joke.split('_____')
        if len(split) > 2:
            continue
        else:
            texta = split[0].strip()
            textb = split[1].strip()
            joke = texta + '_____' + textb

        # replace "" with "
        joke = re.subn('\"\"', '\"', joke)[0]

        # remove 'edits:'
        for edit in edits:
            pat = edit+'.*'
            joke = re.sub(pat, '', joke)

        # replace ..+ with space
        joke = re.subn('\.\..*?', '', joke)[0]

        # GOOD JOKE
        good_jokes.append(joke)
        good_labels.append(labels[ix])

        if labels[ix] == '1':
            pos += 1
        else:
            neg += 1

        assert len(joke.split('_____')) == 2

    assert len(good_jokes) == len(good_labels)

    print('Kept:', len(good_jokes), '/', len(jokes))
    print('percentage:', len(good_jokes) / len(jokes))
    print('pos', pos, 'neg', neg)

    save = name[:-4] + '_clean.tsv'
    with open(os.path.join(data_dir, save), 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter=',')
        for ix, combo in enumerate(zip(good_jokes, good_labels)):
            joke = combo[0]
            label = combo[1]

            writer.writerow([ix, label, 'a', joke])

    #print('Unique', no_dup, no_dup / len(jokes))
