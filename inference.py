import json
from utils.config import *
from sklearn import metrics
from transformers import AutoTokenizer
import torch
import logging
import utils.utils_general as util


from load_data import UDCDataLoader

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

@torch.no_grad()
def inference():
    # Set device
    num_gpus = torch.cuda.device_count()
    use_cuda = num_gpus > 0
    device = torch.device("cuda" if use_cuda else "cpu")

    logging.Formatter.converter = util.kst
    logging.basicConfig(filename=os.path.join(args.model_dir, 'logs_test.txt'),
                        filemode='w', format='%(asctime)s -  %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.ptm_model,
        use_fast=True
    )

    # Build data loader
    udc_dataloader = UDCDataLoader(tokenizer, max_length=args.ptm_seq_size, use_cuda=use_cuda)
    test_loader = udc_dataloader.get_dataloader(
        data=list(json.load(open(args.test_data_dir))),
        batch_size=args.batch_size,
        labels=eval(open(args.label_dir).readlines()[0]),
    )

    # Load model
    print("STARTING TESTING")
    model = torch.load(args.model_dir + '/smp_checkpoint/model.th')
    model.to(device)
    model.eval()
    prediction = []
    total_label = []
    pbar = enumerate(test_loader)
    for i, batch_data in pbar:
        output_state = model(batch_data)
        pred = torch.argmax(output_state, dim=1)
        prediction.extend(pred.cpu().numpy())
        total_label.extend(batch_data["label"].cpu().numpy())

    test_acc = metrics.accuracy_score(total_label, prediction)
    print('Acc:{:.2f}'.format(test_acc))
    logger.info("Test " + "accuracy={:.4f};".format(test_acc))

    # logits_all = np.zeros((4528, args.num_class))
    # for i in range(1):
    #     model = torch.load(args.model_dir + '/model.th')
    #     model.to(device)
    #     model.eval()
    #
    #     labels_all = np.array([], dtype=int)
    #     logits_list = []
    #
    #     for out in test_loader:
    #         utterance, att_mask, token_type_ids, labels = [o.to(device) for o in out]
    #
    #         logits, _, loss = model(utterance, att_mask, labels, stage='test')
    #         logits = logits.data.cpu().numpy()
    #         labels = labels.data.cpu().numpy()
    #         labels_all = np.append(labels_all, labels)
    #         logits_list += list(logits)
    #
    #     test_acc = metrics.accuracy_score(labels_all, np.argmax(np.array(logits_list), 1))
    #     print(f"The test acc of model {i}: ", test_acc)
    #
    #     logits_all += np.array(logits_list)

    # pred_all = np.argmax(logits_all, 1)
    # final_test_acc = metrics.accuracy_score(labels_all, pred_all)
    # print("final test_acc: ", final_test_acc)

    # utils.make_dirs(args.output_dir)
    # output_file = open(os.path.join(args.output_dir, "output.csv"), "w")
    # for i,j in zip(labels_all, pred_all):
    #     print(i,j)
    #     # output_file.write(i + '\t' + j + '\n')
    # output_file.close()

if __name__ == "__main__":
    inference()
