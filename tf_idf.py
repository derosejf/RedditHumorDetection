from nltk.corpus import stopwords
import utils
import csv
from math import log

sw = stopwords.words("english")
doc_frequencies = {}


def load_doc_frequencies(file):
    with open(file, "r") as in_f:
        reader = csv.reader(in_f)
        doc_count = 0
        for row in reader:
            doc_count += 1
            text = row[3].replace("_____", " ")
            words = utils.prepare_text(text).split()
            for word in set(words):
                word = word.lower()
                doc_frequencies[word] = doc_frequencies.setdefault(word, 0) + 1
        for k, v in doc_frequencies.items():
            doc_frequencies[k] = v / doc_count


if __name__ == "__main__":
    load_doc_frequencies("data/train_clean.tsv")
    output_extension = "_tf_idf"
    filepaths = ['data/train_clean.tsv', 'data/dev_clean.tsv', 'data/test_clean.tsv']
    for filepath in filepaths:
        with open(filepath, "r") as in_f:
            with open(filepath.replace(".tsv", f"{output_extension}.tsv"), "w") as out_f:
                reader = csv.reader(in_f)
                writer = csv.writer(out_f)
                for row in reader:
                    scores = []
                    text = row[3].replace("_____", " ")
                    text = utils.prepare_text(text).split()
                    for word in text:
                        if word != "[SEP]":
                            word = word.lower()
                        if word in sw or not word.isalpha() or word not in doc_frequencies.keys():
                            scores.append((word, 0))
                        else:
                            tf = len([w for w in text if w.lower() == word])
                            idf = log(1/doc_frequencies[word])
                            scores.append((word, tf * idf))
                    out_row = row + [scores]
                    writer.writerow(out_row)
