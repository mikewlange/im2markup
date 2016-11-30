import sys, logging, argparse, os

def is_ascii(str):
    try:
        str.decode('ascii')
        return True
    except UnicodeError:
        return False

def process_args(args):
    parser = argparse.ArgumentParser(description='Generate vocabulary file.')

    parser.add_argument('--documents-dir', dest='documents_dir',
                        type=str, required=True,
                        help=('Directory containing processed documents.'
                        ))
    parser.add_argument('--data-path', dest='data_path',
                        type=str, required=True,
                        help=('Input file containing <document_path> <line_idx> per line. This should be the file used for training.'
                        ))
    parser.add_argument('--label-path', dest='label_path',
                        type=str, required=True,
                        help=('Input file containing a tokenized summary per line.'
                        ))
    parser.add_argument('--output-file-source', dest='output_file_source',
                        type=str, required=True,
                        help=('Output file for putting source side vocabulary.'
                        ))
    parser.add_argument('--output-file-target', dest='output_file_target',
                        type=str, required=True,
                        help=('Output file for putting target side vocabulary.'
                        ))
    parser.add_argument('--padding-symbol', dest='padding_symbol',
                        type=str, default='__PAD__',
                        help=('Padding symbol, will be ignored for vocabulary.'
                        ))
    parser.add_argument('--unk-threshold', dest='unk_threshold',
                        type=int, default=1,
                        help=('If the number of occurences of a token is less than (including) the threshold, then it will be excluded from the generated vocabulary.'
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

    documents_dir = parameters.documents_dir
    assert os.path.exists(documents_dir), documents_dir
    label_path = parameters.label_path
    assert os.path.exists(label_path), label_path
    data_path = parameters.data_path
    assert os.path.exists(data_path), data_path
    padding_symbol = parameters.padding_symbol

    formulas = open(label_path).readlines()
    vocab_target = {}
    max_len = 0
    with open(data_path) as fin:
        for line in fin:
            _, line_idx = line.strip().split()
            line_strip = formulas[int(line_idx)].strip()
            tokens = line_strip.split()
            for token in tokens:
                if not is_ascii(token):
                    continue
                assert token != padding_symbol, 'padding_symbol %s occurs in target side!'%padding_symbol
                if token not in vocab_target:
                    vocab_target[token] = 0
                vocab_target[token] += 1

    vocab_target_sort = sorted(list(vocab_target.keys()))
    vocab_target_out = []
    num_unknown = 0
    for word in vocab_target_sort:
        if vocab_target[word] > parameters.unk_threshold:
            vocab_target_out.append(word)
        else:
            num_unknown += 1
    #vocab = ["'"+word.replace('\\','\\\\').replace('\'', '\\\'')+"'" for word in vocab_out]
    vocab_target = [word for word in vocab_target_out]

    with open(parameters.output_file_target, 'w') as fout:
        fout.write('\n'.join(vocab_target))
    logging.info('Target: #vocab: %d, #UNK\'s: %d'%(len(vocab_target), num_unknown))
    vocab_source = {}
    max_len = 0
    with open(data_path) as fin:
        for i, line in enumerate(fin):
            if i % 1000 == 0:
                logging.info('%d lines read...'%i)
            document_path, line_idx = line.strip().split()
            with open(os.path.join(documents_dir, document_path)) as fin:
                for line in fin:
                    line_strip = line.strip()
                    tokens = line_strip.split()
                    for token in tokens:
                        if not is_ascii(token):
                            continue
                        if token == padding_symbol:
                            continue
                        if token not in vocab_source:
                            vocab_source[token] = 0
                        vocab_source[token] += 1

    vocab_source_sort = sorted(list(vocab_source.keys()))
    vocab_source_out = []
    num_unknown = 0
    for word in vocab_source_sort:
        if vocab_source[word] > parameters.unk_threshold:
            vocab_source_out.append(word)
        else:
            num_unknown += 1
    #vocab = ["'"+word.replace('\\','\\\\').replace('\'', '\\\'')+"'" for word in vocab_out]
    vocab_source = [word for word in vocab_source_out]

    with open(parameters.output_file_source, 'w') as fout:
        fout.write('\n'.join(vocab_source))
    logging.info('Source: #vocab: %d, #UNK\'s: %d'%(len(vocab_source), num_unknown))

if __name__ == '__main__':
    main(sys.argv[1:])
    logging.info('Jobs finished')
