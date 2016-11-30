import sys, logging, argparse, os, h5py
import numpy as np

def process_args(args):
    parser = argparse.ArgumentParser(description='Generate vocabulary file.')

    parser.add_argument('--word2vec', dest='word2vec',
                        type=str, required=True,
                        help=('word2vec path containing embeddings.'
                        ))
    parser.add_argument('--vocab-file', dest='vocab_file',
                        type=str, required=True,
                        help=('Source side vocabulary.'
                        ))
    parser.add_argument('--output-file', dest='output_file',
                        type=str, required=True,
                        help=('Output file for putting word embeddings.'
                        ))
    parser.add_argument('--offset', dest='offset',
                        type=int, default=1,
                        help=('Source side vocabulary offset. Default: reserve the first token for padding_symbol'
                        ))
    parser.add_argument('--log-path', dest="log_path",
                        type=str, default='log.txt',
                        help=('Log file path, default=log.txt' 
                        ))
    parameters = parser.parse_args(args)
    return parameters

def build_embeds(word2vec, output_file, vocab, offset=0):
    def load_bin_vec(word2vec, vocab):
        """
        Loads 300x1 word vecs from Google (Mikolov) word2vec
        """
        word_vecs = {}
        with open(word2vec, "rb") as f:
            header = f.readline()
            vocab_size, layer1_size = map(int, header.split())
            binary_len = np.dtype('float32').itemsize * layer1_size
            for line in xrange(vocab_size):
                word = []
                while True:
                    ch = f.read(1)
                    if ch == ' ':
                        word = ''.join(word)
                        break
                    if ch != '\n':
                        word.append(ch)
                if word in vocab:
                    word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')
                else:
                    f.read(binary_len)
        return word_vecs

    word_vecs = load_bin_vec(word2vec, vocab)
    embeds = np.random.uniform(-0.25, 0.25, (len(vocab)+offset, len(word_vecs.values()[0])))
    for word, vec in word_vecs.iteritems():
        embeds[vocab[word]] = vec

    f = h5py.File(output_file, "w")
    f["word_vecs"] = np.array(embeds)
    f.close()

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

    word2vec = parameters.word2vec
    assert os.path.exists(word2vec), word2vec
    vocab_file = parameters.vocab_file
    assert os.path.exists(vocab_file), vocab_file
    output_file = parameters.output_file
    offset = parameters.offset

    vocab = {}
    idx = offset
    with open(vocab_file) as fin:
        for line in fin:
            vocab[line.strip()] = idx
            idx += 1

    build_embeds(word2vec, output_file, vocab, offset)

if __name__ == '__main__':
    main(sys.argv[1:])
    logging.info('Jobs finished')
