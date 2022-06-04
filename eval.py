import json
import logging
from transformers import AutoTokenizer, set_seed

import utils.utils_general as util
from load_data import UDCDataLoader
from model.modules import BilstmEncoder, BertEncoder
from model.kudc import Trainer
from utils.config import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def train():
    # Set logger & seed
    set_seed(args.seed)
    util.make_dirs(args.model_dir)
    logging.Formatter.converter = util.kst
    logging.basicConfig(filename=os.path.join(args.model_dir, 'logs_train.txt'),
                        filemode='w', format='%(asctime)s -  %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info(args)

    # Set device
    num_gpus = torch.cuda.device_count()
    use_cuda = num_gpus > 0
    device = torch.device("cuda" if use_cuda else "cpu")

    logger.info(f"***** using {device} *****")
    logger.info(f"***** num GPU: {num_gpus} *****")

    # Build PLM config & tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.ptm_model,
        use_fast=True
    )
    logger.info(tokenizer)

    # Build data loader
    udc_dataloader = UDCDataLoader(tokenizer, max_length=args.ptm_seq_size, use_cuda=use_cuda)
    train_loader = udc_dataloader.get_dataloader(
        data=list(json.load(open(args.train_data_dir))),
        batch_size=args.batch_size,
        labels=eval(open(args.label_dir).readlines()[0]),
    )
    valid_loader = udc_dataloader.get_dataloader(
        data=list(json.load(open(args.dev_data_dir))),
        batch_size=args.batch_size,
        labels=eval(open(args.label_dir).readlines()[0]),
    )

    # Build model
    myModel = BilstmEncoder(
        tokenizer.vocab_size,
        args.bert_hidden_size,
        args.rnn_hidden_size,
        args.rnn_dropout,
        args.ptm_seq_size,
        args.num_label,
        args.batch_size)


    # Build trainer
    total_train_steps = len(train_loader) * args.epoch
    num_warmup_steps = int(args.warmup * total_train_steps)

    trainer = Trainer(model=myModel,
                      tokenizer=tokenizer,
                      device=device,
                      dataset_name='smp',
                      model_name='bilstm',
                      lr=args.lr,
                      num_train_steps=total_train_steps,
                      num_warmup_steps=num_warmup_steps,
                      logger=logger)

    trainer.train(train_loader, epoch=args.epoch, eval_data=valid_loader)

if __name__ == '__main__':
    train()

