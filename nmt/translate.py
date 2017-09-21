from __future__ import print_function
from __future__ import division

import os
import time
from itertools import izip
from codecs import open

import numpy
import tensorflow as tf

import nmt.all_constants as ac
import nmt.utils as ut
from nmt.model import Model
from nmt.data_manager import DataManager
import nmt.configurations as configurations

class Translator(object):
    def __init__(self, args):
        super(Translator, self).__init__()
        self.config = getattr(configurations, args.proto)()
        self.reverse = self.config['reverse']
        self.unk_repl = self.config['unk_repl']
        self.logger = ut.get_logger(self.config['log_file'])

        self.input_file = args.input_file
        self.model_file = args.model_file
        self.plot_align = args.plot_align

        if self.input_file is None or self.model_file is None or not os.path.exists(self.input_file) or not os.path.exists(self.model_file + '.meta'):
            raise ValueError('Input file or model file does not exist')

        self.data_manager = DataManager(self.config)
        _, self.src_ivocab = self.data_manager.init_vocab(self.data_manager.src_lang)
        _, self.trg_ivocab = self.data_manager.init_vocab(self.data_manager.trg_lang)
        self.translate()

    def get_model(self, mode):
        reuse = mode != ac.TRAINING
        d = self.config['init_range']
        initializer = tf.random_uniform_initializer(-d, d)
        with tf.variable_scope(self.config['model_name'], reuse=reuse, initializer=initializer):
            return Model(self.config, mode)

    def translate(self):
        with tf.Graph().as_default():
            train_model = self.get_model(ac.TRAINING)
            model = self.get_model(ac.VALIDATING)

            with tf.Session() as sess:
                self.logger.info('Restore model from {}'.format(self.model_file))
                saver = tf.train.Saver(var_list=tf.trainable_variables())
                saver.restore(sess, self.model_file)

                best_trans_file = self.input_file + '.best_trans'
                beam_trans_file = self.input_file + '.beam_trans'
                open(best_trans_file, 'w').close()
                open(beam_trans_file, 'w').close()
                ftrans = open(best_trans_file, 'w', 'utf-8')
                btrans = open(beam_trans_file, 'w', 'utf-8')

                self.logger.info('Start translating {}'.format(self.input_file))
                start = time.time()
                count = 0
                for (src_input, src_seq_len, no_unk_src_toks) in self.data_manager.get_trans_input(self.input_file):
                    feed = {
                        model.src_inputs: src_input,
                        model.src_seq_lengths: src_seq_len
                    }
                    probs, scores, symbols, parents, alignments = sess.run([model.probs, model.scores, model.symbols, model.parents, model.alignments], feed_dict=feed)
                    alignments = numpy.transpose(alignments, axes=(1, 0, 2))

                    probs = numpy.transpose(numpy.array(probs))
                    scores = numpy.transpose(numpy.array(scores))
                    symbols = numpy.transpose(numpy.array(symbols))
                    parents = numpy.transpose(numpy.array(parents))

                    best_trans, best_trans_ids, beam_trans, best_trans_alignments = ut.get_trans(probs, scores, symbols, parents, alignments, no_unk_src_toks, self.trg_ivocab, reverse=self.reverse, unk_repl=self.unk_repl)
                    best_trans_wo_eos = best_trans.split()[:-1]
                    best_trans_wo_eos = u' '.join(best_trans_wo_eos)
                    ftrans.write(best_trans_wo_eos + '\n')
                    btrans.write(beam_trans + '\n\n')

                    if self.plot_align:
                        src_input = numpy.reshape(src_input, [-1])
                        if self.reverse:
                            src_input = src_input[::-1]
                            no_unk_src_toks = no_unk_src_toks[::-1]
                        trans_toks = best_trans.split()
                        best_trans_alignments = numpy.array(best_trans_alignments)[:len(trans_toks)]
                        filename = '{}_{}.png'.format(self.input_file, count)

                        ut.plot_head_map(best_trans_alignments, trans_toks, best_tran_ids, no_unk_src_toks, src_input, filename)

                    count += 1
                    if count % 100 == 0:
                        self.logger.info('  Translating line {}, average {} seconds/sent'.format(count, (time.time() - start) / count))

                ftrans.close()
                btrans.close()

                self.logger.info('Done translating {}, it takes {} minutes'.format(self.input_file, float(time.time() - start) / 60.0))


