#!/usr/bin/env python
# Preprocess images for ease of training
import sys, os, argparse, json, glob, logging
import numpy as np
sys.path.insert(0, '%s'%os.path.join(os.path.dirname(__file__), '../utils/'))
from image_utils import *
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool 

def process_args(args):
    parser = argparse.ArgumentParser(description='Process documents for ease of training. For each document, we see which bucket it falls into and pad them with __PAD__ to match the smallest bucket that can hold it.')

    parser.add_argument('--input-dir', dest='input_dir',
                        type=str, required=True,
                        help=('Input directory containing orginal documents.'
                        ))
    parser.add_argument('--output-dir', dest='output_dir',
                        type=str, required=True,
                        help=('Output directory to put processed documents.'
                        ))
    parser.add_argument('--num-threads', dest='num_threads',
                        type=int, default=4,
                        help=('Number of threads, default=4.'
                        ))
    parser.add_argument('--buckets', dest='buckets',
                        type=str, default='[[10, 150], [20, 150], [30, 150], [40, 150], [40, 150], [50, 150], [100, 150], [50, 200], [50, 300]]',
                        help=('Bucket sizes used for grouping. Should be a Json string.'
                        ))
    parser.add_argument('--one-line-mode', dest='one_line_mode', action='store_true',
                        help=('One line mode flag, if set, then all sentences will be concatenated to one line.'
                        ))
    parser.add_argument('--no-one-line-mode', dest='one_line_mode', action='store_false')
    parser.set_defaults(one_line_mode=False)
    parser.add_argument('--truncate-large-document', dest='truncate_large_document', action='store_true',
                        help=('If set, then documents larger than the largest bin size (either with more sentences or with longer sentences) will be truncated. Otherwise, such documents will be left as-is (But will be appended with EOS and padded).'
                        ))
    parser.add_argument('--no-truncate-large-document', dest='truncate_large_document', action='store_false')
    parser.set_defaults(truncate_large_document=True)
    parser.add_argument('--padding-symbol', dest='padding_symbol',
                        type=str, default='__PAD__',
                        help=('Padding symbol, will be ignored for vocabulary.'
                        ))
    parser.add_argument('--eos-symbol', dest='eos_symbol',
                        type=str, default='__EOS__',
                        help=('End of sentence symbol, will be appended to the end of each sentence.'
                        ))
    parser.add_argument('--log-path', dest="log_path",
                        type=str, default='log.txt',
                        help=('Log file path, default=log.txt' 
                        ))
    parser.add_argument('--postfix', dest='postfix',
                        type=str, default='.txt',
                        help=('The format of documents, default=".txt". We use it to search in input-dir.'
                        ))
    parameters = parser.parse_args(args)
    return parameters

def main_parallel(l):
    filename, postfix, output_filename, buckets, padding_symbol, eos_symbol, one_line_mode, truncate_large_document = l
    postfix_length = len(postfix)
    with open(filename) as fin:
        with open(output_filename, 'w') as fout:
            sentences = fin.readlines()
            if one_line_mode:
                sentences = [' '.join([sentence.strip() for sentence in sentences])]
            sentence_length = len(sentences)
            word_length = -1
            for sentence in sentences:
                word_length = max(word_length, 1+len(sentence.strip().split()))
            j = -1
            for i in range(len(buckets)):
                if sentence_length <= buckets[i][0] and word_length <= buckets[i][1]:
                    j = i
                    break
            if j < 0:
                if truncate_large_document:
                    logging.info('Warning: %s is too large to put into any buckets, will be truncated!'%filename)
                    final_sentence_length = buckets[len(buckets)-1][0]
                    final_word_length = buckets[len(buckets)-1][1]
                else:
                    logging.info('Warning: %s is too large to put into any buckets, will be left as-is!'%filename)
                    final_sentence_length = sentence_length
                    final_word_length = word_length
            else:
                final_sentence_length = buckets[i][0]
                final_word_length = buckets[i][1]
            for i in range(final_sentence_length):
                if i < len(sentences):
                    words = sentences[i].strip().split() + [eos_symbol]
                else:
                    words = []
                words_out = []
                for j in range(final_word_length):
                    if j < len(words):
                        word = words[j]
                    else:
                        word = padding_symbol
                    words_out.append(word)
                fout.write(' '.join(words_out)+'\n')
                    
                    

def main(args):
    parameters = process_args(args)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)-15s %(name)-5s %(levelname)-8s %(message)s',
        filename=parameters.log_path)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)-15s %(name)-5s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

    logging.info('Script being executed: %s'%__file__)

    output_dir = parameters.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    input_dir = parameters.input_dir
    postfix = parameters.postfix
    buckets = json.loads(parameters.buckets)
    padding_symbol = parameters.padding_symbol
    eos_symbol = parameters.eos_symbol
    one_line_mode = parameters.one_line_mode
    truncate_large_document = parameters.truncate_large_document

    filenames = glob.glob(os.path.join(input_dir, '*'+postfix))
    logging.info('Creating pool with %d threads'%parameters.num_threads)
    pool = ThreadPool(parameters.num_threads)
    logging.info('Jobs running...')
    results = pool.map(main_parallel, [(filename, postfix, os.path.join(output_dir, os.path.basename(filename)), buckets, padding_symbol, eos_symbol, one_line_mode, truncate_large_document) for filename in filenames])
    pool.close() 
    pool.join() 

if __name__ == '__main__':
    main(sys.argv[1:])
    logging.info('Jobs finished')
