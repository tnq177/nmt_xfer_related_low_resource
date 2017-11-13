Tested with Python 2.7.3 and Tensorflow 1.1
## What does it do exactly?
It trains a global attentional NMT model based on [Effective Approaches to Attention-based Neural Machine Translation](https://arxiv.org/abs/1508.04025) using beam search with length normalization by [Google's Neural Machine Translation System: Bridging the Gap between Human and Machine Translation](https://arxiv.org/abs/1609.08144).  

It trains with negative loglikelihood objective and periodically evaluates the performance (BLEU score) on dev set. It can save a few best models as defined in the configuration function. I always save one best model (highest BLEU score on the dev set). 

## How to use this code?
* Put your data including ```{train,dev,test}.{source,target}``` into a folder, let's call it ```s2t```, in the ```nmt/data``` directory
* Write corresponding configuration function in ```configurations.py```, let's call it ```s2t_config```
* To train, run ```python -m nmt --proto s2t_config```
* To translate (with unk replacement) a file with saved model, run ```python -m nmt --mode translate --unk-repl --proto s2t_config --model-file nmt/saved_models/your_saved_model_dir_name/your_model_name-best-bleu-score.cpkt --input-file path_to_file_to_translate```

An example configuration function ```test_en2vi``` is provided.

## How to train from a pretrained model?
There's a ```reload``` option. If this is set to ```True```, and if the code sees a corresponding checkpoint it will automatically start training from that checkpoint.  

## How to freeze some parameters during training?
For example, if we want to not train the output embedding, we can access this from our model as ```model.softmax.logit_layer.W```, so we fix it this way:

```python -m nmt --proto s2t --fixed-var-list softmax.logit_layer.W```

## References
I was inspired by code from lots of examples out there I can't start to remember. A lot of code is referenced from [Blocks examples](https://github.com/mila-udem/blocks-examples), [DL4MT](https://github.com/nyu-dl/dl4mt-tutorial), and ```multi-bleu.perl``` is taken from [Moses](https://github.com/moses-smt/mosesdecoder).


We use this code for the paper [Transfer Learning across Low-Resource, Related Languages for Neural Machine Translation](https://arxiv.org/abs/1708.09803) which will appear at IJCNLP'17. We use the code at [subword-nmt](https://github.com/rsennrich/subword-nmt) for processing BPE as explained in the paper.