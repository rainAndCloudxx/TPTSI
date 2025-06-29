from yacs.config import CfgNode as CN

_C = CN()

"""
    data settings
"""
_C.Dataset = 'MOH'
_C.seed = 24

_C.MODEL = CN()
_C.MODEL.num_classes = 2
# embedding dim
_C.MODEL.embed_dim = 768
# use the average of the first and the last hidden layer of PLMs as word embeddings
_C.MODEL.first_last_avg = True
# drop out rate
_C.MODEL.dropout = 0.2
# number of attention heads
_C.MODEL.RobertaPath = './roberta-base'


_C.TRAIN = CN()
_C.TRAIN.lr = 1e-5
# the directory to save the training logs
_C.TRAIN.output = './data/logs'



_C.MOH = CN()
_C.MOH.BatchSize = 16
_C.MOH.max_length = 16
_C.MOH.epochs = 20
_C.MOH.data_dir = './data'
_C.MOH.savaPath = './result/MOH_Weights.pth'

_C.TroFi = CN()
_C.TroFi.BatchSize = 16
_C.TroFi.max_length = 64
_C.TroFi.epochs = 10
_C.TroFi.data_dir = './data'
_C.TroFi.savaPath = './result/TroFi_Weights.pth'

_C.VUA = CN()
_C.VUA.BatchSize = 128
_C.VUA.max_length = 64
_C.VUA.epochs = 10
_C.VUA.data_dir = './data'
_C.VUA.savaPath = './result/VUA_Weights.pth'
# tokens separated by these punctuations can mark a context
# _C.DATA.sep_puncs = [',', ' ,', '?', ' ?', ';', ' ;', '.', ' .', '!', ' !', '</s>', '[SEP]']
# # whether to use pos tag
# _C.DATA.use_pos = False
# # whether to use example sentence to trigger target word basic meaning
# _C.DATA.use_eg_sent = True
# # whether to use context feature
# _C.DATA.use_context = True
# _C.DATA.use_sim = True
# _C.DATA.use_ex = True
# # the pretrained language model to use. Please pre-download. default is RoBERTa.
# _C.DATA.plm = './roberta-base'

# _C.DATA = CN()
# _C.DATA.data_dir = './data'
# # tokens separated by these punctuations can mark a context
# _C.DATA.sep_puncs = [',', ' ,', '?', ' ?', ';', ' ;', '.', ' .', '!', ' !', '</s>', '[SEP]']
# # whether to use pos tag
# _C.DATA.use_pos = False
# # whether to use example sentence to trigger target word basic meaning
# _C.DATA.use_eg_sent = True
# # whether to use context feature
# _C.DATA.use_context = True
# _C.DATA.use_sim = True
# _C.DATA.use_ex = True
# # the pretrained language model to use. Please pre-download. default is RoBERTa.
# _C.DATA.plm = './roberta-base'

"""
    model settings
"""
# _C.MODEL = CN()
# _C.MODEL.num_classes = 2
# # embedding dim
# _C.MODEL.embed_dim = 768
# # use the average of the first and the last hidden layer of PLMs as word embeddings
# _C.MODEL.first_last_avg = True
# # drop out rate
# _C.MODEL.dropout = 0.2
# # number of attention heads
# _C.MODEL.num_heads = 12

# '''
#     training settings
# '''
# _C.TRAIN = CN()
# _C.TRAIN.lr = 1e-5
# # the directory to save the training logs
# _C.TRAIN.output = './data/logs'

# _C.gpu = '0'
# _C.seed = 24
# # do eval only
# _C.eval_mode = False
# _C.log = 'log_test'
# _C.cl = False


def update_config(config, args):
    config.defrost()

    print('=> merge config from {}'.format(args.cfg))
    config.merge_from_file(args.cfg)

    if args.gpu:
        config.gpu = args.gpu

    if args.eval:
        config.eval_mode = True

    if args.log:
        config.log = args.log

    config.freeze()


def get_config(args):
    """Get a yacs CfgNode object with default values."""
    config = _C.clone()
    update_config(config, args)

    return config
