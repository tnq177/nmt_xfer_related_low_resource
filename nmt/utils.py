from __future__ import print_function
from __future__ import division

import os
import logging
from datetime import timedelta

import numpy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import tensorflow as tf
import subprocess

import nmt.all_constants as ac

## TRAINING UTILS
def get_logger(logfile=None):
    _logfile = logfile if logfile else './DEBUG.log'
    """Global logger for every logging"""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s:%(filename)s:%(lineno)s - %(funcName)20s(): %(message)s')

    if not logger.handlers:
        debug_handler = logging.FileHandler(_logfile)
        debug_handler.setFormatter(formatter)
        debug_handler.setLevel(logging.DEBUG)
        logger.addHandler(debug_handler)

    return logger


def shuffle_file(input_file):
    shuffled_file = input_file + '.shuf'
    commands = 'bash ./scripts/shuffle_file.sh {} {}'.format(input_file, shuffled_file)
    subprocess.check_call(commands, shell=True)
    subprocess.check_call('mv {} {}'.format(shuffled_file, input_file), shell=True)


def get_validation_frequency(train_length_file, val_frequency, batch_size):
    if val_frequency > 1.0:
        return val_frequency
    else:
        with open(train_length_file) as f:
            line = f.readline().strip()
            num_train_sents = int(line)

        return int((num_train_sents * val_frequency) // batch_size)

def format_seconds(seconds):
    return str(timedelta(seconds=seconds))

## MODEL UTILS
def get_lstm_cell(scope, num_layers, rnn_size, output_keep_prob=1.0, seed=42, reuse=False):
    def get_cell(_rnn_size, _output_keep_prob, _reuse, _seed):
        cell = tf.contrib.rnn.LSTMCell(_rnn_size, state_is_tuple=True, reuse=_reuse)
        return tf.contrib.rnn.DropoutWrapper(
            cell,
            output_keep_prob=_output_keep_prob,
            dtype=tf.float32,
            seed=_seed)

    with tf.variable_scope(scope):
        if num_layers <= 1:
            return get_cell(rnn_size, output_keep_prob, reuse, seed)
        else:
            cells = []
            for i in xrange(num_layers):
                cell = get_cell(rnn_size, output_keep_prob, reuse, seed)
                cells.append(cell)

            return tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)


def tensor_to_lstm_state(state_tensor, num_layers):
    if num_layers == 1:
        return tf.contrib.rnn.LSTMStateTuple(state_tensor[0, :, :], state_tensor[1, :, :])
    else:
        state_list = []
        for k in xrange(num_layers):
            c_m = state_tensor[k, :, :, :]
            state_list.append(tf.contrib.rnn.LSTMStateTuple(c_m[0, :, :], c_m[1, :, :]))

        return tuple(state_list)


## VAL UTILS
def ids_to_trans(trans_ids, trans_alignments, no_unk_src_toks, ivocab, unk_repl=True):
    words = []
    word_ids = []

    # Could have done better but this is clearer to me
    if unk_repl:
        for idx, word_idx in enumerate(trans_ids):
            if word_idx == ac.UNK_ID:
                # Replace UNK with higest attention source words
                alignment = trans_alignments[idx]
                highest_att_src_tok_pos = numpy.argmax(alignment)
                words.append(no_unk_src_toks[highest_att_src_tok_pos])
            else:
                words.append(ivocab[word_idx])
            word_ids.append(word_idx)

            if word_idx == ac.EOS_ID:
                break
    else:
        for idx, word_idx in enumerate(trans_ids):
            words.append(ivocab[word_idx])
            word_ids.append(word_idx)
            if word_idx == ac.EOS_ID:
                break

    return u' '.join(words), word_ids

def get_trans(probs, scores, symbols, parents, alignments, no_unk_src_toks, ivocab, reverse=True, unk_repl=True):
    sorted_rows = numpy.argsort(scores[:, -1])[::-1]
    best_trans_alignments = []
    best_trans = None
    best_tran_ids = None
    beam_trans = []
    for i, r in enumerate(sorted_rows):
        row_idx = r
        col_idx = scores.shape[1] - 1

        trans_ids = []
        trans_alignments = []
        while True:
            if col_idx < 0:
                break

            trans_ids.append(symbols[row_idx, col_idx])
            align = alignments[row_idx, col_idx, :]
            trans_alignments.append(align)

            if i == 0:
                best_trans_alignments.append(align if not reverse else align[::-1])

            row_idx = parents[row_idx, col_idx]
            col_idx -= 1

        trans_ids = trans_ids[::-1]
        trans_alignments = trans_alignments[::-1]
        trans_out, trans_out_ids = ids_to_trans(trans_ids, trans_alignments, no_unk_src_toks, ivocab, unk_repl=unk_repl)
        beam_trans.append(u'{} {:.2f} {:.2f}'.format(trans_out, scores[r, -1], probs[r, -1]))
        if i == 0: # highest prob trans
            best_trans = trans_out
            best_tran_ids = trans_out_ids

    return best_trans, best_tran_ids, u'\n'.join(beam_trans), best_trans_alignments[::-1]

def plot_head_map(mma, target_labels, target_ids, source_labels, source_ids, filename):
    """https://github.com/EdinburghNLP/nematus/blob/master/utils/plot_heatmap.py
    Change the font in family param below. If the system font is not used, delete matplotlib 
    font cache https://github.com/matplotlib/matplotlib/issues/3590
    """
    fig, ax = plt.subplots()
    heatmap = ax.pcolor(mma, cmap=plt.cm.Blues)

    # put the major ticks at the middle of each cell
    ax.set_xticks(numpy.arange(mma.shape[1]) + 0.5, minor=False)
    ax.set_yticks(numpy.arange(mma.shape[0]) + 0.5, minor=False)

    # without this I get some extra columns rows
    # http://stackoverflow.com/questions/31601351/why-does-this-matplotlib-heatmap-have-an-extra-blank-column
    ax.set_xlim(0, int(mma.shape[1]))
    ax.set_ylim(0, int(mma.shape[0]))

    # want a more natural, table-like display
    ax.invert_yaxis()
    ax.xaxis.tick_top()

    # source words -> column labels
    ax.set_xticklabels(source_labels, minor=False, family='Source Code Pro')
    for xtick, idx in zip(ax.get_xticklabels(), source_ids):
        if idx == ac.UNK_ID:
            xtick.set_color('b')
    # target words -> row labels
    ax.set_yticklabels(target_labels, minor=False, family='Source Code Pro')
    for ytick, idx in zip(ax.get_yticklabels(), target_ids):
        if idx == ac.UNK_ID:
            ytick.set_color('b')

    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig(filename)
    plt.close('all')