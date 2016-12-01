#!/usr/bin/env python
import sys, os, argparse, logging
import numpy as np
import PIL
from PIL import Image

def process_args(args):
    parser = argparse.ArgumentParser(description='Process train, test, development files (<document_path> <label_idx>) for formatting files such that can be used for training. (<document_path> <label_idx>). Additionaly, if <filter> flag is set, large documents, too long target summaries will be discarded.')

    parser.add_argument('--documents-dir', dest='documents_dir',
                        type=str, default='',
                        help=('Directory containing processed documents'
                        ))
    parser.add_argument('--data-path', dest='data_path',
                        type=str, required=True,
                        help=('Input file path containing <document_path> <label_idx> per line.'
                        ))
    parser.add_argument('--output-path', dest='output_path',
                        type=str, required=True,
                        help=('Output file path containing <document_path> <label_idx> per line. If filter flag is set, then the output file may have less lines than original file.'
                        ))

    parser.add_argument('--label-path', dest='label_path',
                        type=str, default='',
                        help=('Input label path containing <summary> per line. This is required if filter flag is set, and instances with blank summaries will be discarded.'
                        ))
    parser.add_argument('--filter', dest='filter', action='store_true',
                        help=('Filter flag, if set, then too large documents, target summaries that have too many tokens or are blank will be discarded.'
                        ))
    parser.add_argument('--no-filter', dest='filter', action='store_false')
    parser.set_defaults(filter=False)
    parser.add_argument('--max-num-sentences', dest='max_num_sentences',
                        type=str, default=100,
                        help=('If filter flag is set, documents with more than max-num-sentences sentences will be discarded in the output file.'
                        ))
    parser.add_argument('--max-sentence-length', dest='max_sentence_length',
                        type=str, default=300,
                        help=('If filter flag is set, documents with longer sentence length than max-sentence-length will be discarded in the output file.'
                        ))
    parser.add_argument('--max-summary-length', dest='max_summary_length',
                        type=str, default=150,
                        help=('If filter flag is set, documents with more than max-summary-length target summary tokens will be discarded in the output file.'
                        ))
    parser.add_argument('--log-path', dest="log_path",
                        type=str, default='log.txt',
                        help=('Log file path, default=log.txt' 
                        ))
    parameters = parser.parse_args(args)
    return parameters

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
    data_path = parameters.data_path
    output_path = parameters.output_path
    documents_dir = parameters.documents_dir

    num_discard = 0
    num_nonexist = 0

    if parameters.filter:
        assert os.path.isfile(parameters.label_path), parameters.label_path
        labels = open(parameters.label_path).readlines()
    with open(output_path, 'w') as fout:
        with open(data_path, 'r') as fdata:
            for num_lines, line in enumerate(fdata):
                if num_lines % 1000 == 0:
                    logging.info('%d lines processed'%num_lines)
                line_strip = line.strip()
                if len(line_strip) > 0:
                    doc_path, line_idx = line_strip.split()
                    doc_path = os.path.join(documents_dir, doc_path)
                    if parameters.filter:
                        if not os.path.exists(doc_path):
                            logging.warning('%s does not exist!'%os.path.basename(doc_path))
                            num_nonexist += 1
                            continue
                        with open(doc_path) as fdoc:
                            lines = fdoc.readlines()
                        w = len(lines[0].split())
                        h = len(lines)
                    else:
                        w = 0
                        h = 0
                    if (not parameters.filter) or (w <= parameters.max_sentence_length and h <= parameters.max_num_sentences):
                        if parameters.filter:
                            label = labels[int(line_idx)]
                            if len(label.strip()) == 0:
                                logging.info('%s discarded due to cannot-be-parsed formula!'%os.path.basename(doc_path))
                                continue
                            if len(label.strip().split()) > parameters.max_summary_length:
                                logging.info('%s discarded due to too many summary tokens!'%os.path.basename(doc_path))
                                continue
                        fout.write('%s %s\n'%(os.path.basename(doc_path),line_idx))
                    else:
                        logging.info('%s discarded due to large image size!'%os.path.basename(doc_path))
                        num_discard += 1
    logging.info('%d discarded. %d not found in %s.'%(num_discard, num_nonexist, documents_dir))


if __name__ == '__main__':
    main(sys.argv[1:])
    logging.info('Jobs finished')
