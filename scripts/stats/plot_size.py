import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys

buckets = [(10,150), (20,150), (30,150), (40,150), (50,150), (100,150), (50,200), (50,300)]
if __name__ == '__main__':
    if len(sys.argv) != 3:
        print >> sys.stderr, 'Usage: python %s <input-file> <output-file>'%sys.argv[0]
        sys.exit(1)
    x, y = [], []
    bins = [0 for bucket in buckets]
    num_total, num_keep = 0, 0
    with open(sys.argv[1]) as fin:
        for line in fin:
            sentences = line.strip('</s>').split('</s>')
            sentence_length = len(sentences)
            word_length = -1
            for sentence in sentences:
                word_length = max(word_length, len(sentence.split()))
            x.append(sentence_length)
            y.append(word_length)
            j = -1
            for i in range(len(bins)):
                if sentence_length <= buckets[i][0] and word_length <= buckets[i][1]:
                    j = i
                    break
            if j >= 0:
                bins[j] += 1
                num_keep += 1
            num_total += 1
        print ('#total: %d, #keep:%d'%(num_total, num_keep))
        print (bins)
        plt.scatter(x, y)
        plt.savefig(sys.argv[2])
