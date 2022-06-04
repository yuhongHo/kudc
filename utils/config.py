import os
import argparse
import torch

if (torch.cuda.device_count() > 0):
    USE_CUDA = True
else:
    USE_CUDA = False

parser = argparse.ArgumentParser(description='Knowledge-based UDC')
parser.add_argument('-ptm', '--ptm_model', help='pretrained model', default='bert-base-chinese')
parser.add_argument('-bsz','--batch_size', help='Batch_size', default=32)
parser.add_argument('-e', '--epoch', help='number of epochs to train/valid', default=20)
parser.add_argument('-lr', '--lr', help='learning rate', default=1e-5)
parser.add_argument('-wp','--warmup', help="Proportion of lr increasing steps", default=0.1)
parser.add_argument('-sd', '--seed', help='random seed', default=1337)
parser.add_argument('-psz', '--ptm_seq_size', help='pretrained model sequence size', default=128)
parser.add_argument('-rnndr', '--rnn_dropout', help='the dropout of bilstm', required=False)
parser.add_argument('-rhsz', '--rnn_hidden_size', help='hidden size in bilstm', default=150)
parser.add_argument('-bhsz', '--bert_hidden_size', help='hidden size in bilstm', default=768)
parser.add_argument('-nl', '--num_label', help='number of label', type=int, default=31)
parser.add_argument('-c', '--clip', help='clip value', default=0.5)

# Container environment
parser.add_argument('-mdir', '--model_dir', help='path to save output', default=os.environ.get('MODEL_DIR', './output'))
parser.add_argument('-train_dir', '--train_data_dir', help='path to training data', default=os.environ.get('TRAIN_DATA_DIR', './data/smp/0_train.txt'))
parser.add_argument('-dev_dir', '--dev_data_dir', help='path to validation data', default=os.environ.get('DEV_DATA_DIR', './data/smp/0_dev.txt'))
parser.add_argument('-test_dir', '--test_data_dir', help='path to test data', default=os.environ.get('TEST_DATA_DIR', './data/smp/test.txt'))
parser.add_argument('-label_dir', '--label_dir', help='path to label file', default=os.environ.get('LABEL_DIR', './data/smp/labels.txt'))

# args = vars(parser.parse_args())
args = parser.parse_args()
print(str(args))
print("USE_CUDA: "+str(USE_CUDA))