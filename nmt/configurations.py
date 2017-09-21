from __future__ import print_function
from __future__ import division

import os
import nmt.all_constants as ac

def test_en2vi():
    config = {}

    config['model_name']        = 'test_en2vi'
    config['save_to']           = './nmt/saved_models/{}'.format(config['model_name'])
    config['src_lang']          = 'en'
    config['trg_lang']          = 'vi'
    config['data_dir']          = './nmt/data/test_en2vi'
    config['log_file']          = './nmt/DEBUG.log'
    config['rnn_type']          = ac.LSTM
    config['batch_size']        = 32
    config['num_layers']        = 2
    config['enc_rnn_size']      = 512
    config['dec_rnn_size']      = 512
    config['src_embed_size']    = 512
    config['trg_embed_size']    = 512
    config['max_src_length']    = 50
    config['max_trg_length']    = 50
    config['init_range']        = 0.01
    config['max_epochs']        = 5
    config['lr']                = 1.0
    config['lr_decay']          = 0.5
    config['optimizer']         = ac.ADADELTA
    config['input_keep_prob']   = 0.8
    config['output_keep_prob']  = 0.8
    config['src_vocab_size']    = 17700
    config['trg_vocab_size']    = 7700
    config['grad_clip']         = 5.0
    config['reverse']           = True
    config['score_func_type']   = ac.SCORE_FUNC_GEN
    config['feed_input']        = True
    config['unk_repl']          = False
    config['reload']            = True
    config['validate_freq']     = 1.0
    config['save_freq']         = 100
    config['beam_size']         = 12
    config['beam_alpha']        = 0.8
    config['n_best']            = 1
    config['val_trans_out']     = os.path.join(config['save_to'], 'validation_trans.txt')
    config['val_beam_out']      = os.path.join(config['save_to'], 'beam_trans.txt')

    return config