from sklearn import metrics
from torch.nn import CrossEntropyLoss
from torch import optim
from torch.optim import lr_scheduler
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup, AdamW
from utils.config import *

class Trainer(object):
    def __init__(self, model, tokenizer, device, dataset_name, logger, model_name,
                 lr, num_train_steps, num_warmup_steps, clip=0.5, output_dir=None, global_step=0, reset=True):

        self.model = model.to(device)
        self.loss = CrossEntropyLoss()
        self.tokenizer = tokenizer
        self.logger = logger
        self.dataset_name = dataset_name
        self.clip = clip
        self.output_dir = output_dir
        self.lr = lr

        # Initialize optimizers and criterion
        self.optimizer, self.scheduler = self.build_adam_optimizer(model_name,
                                                                   lr,
                                                                   num_train_steps,
                                                                   num_warmup_steps,
                                                                   global_step)
        if reset:
            self.reset()

    def reset(self):
        self.train_loss = 0.0
        self.best_score = 0.0

    def build_adam_optimizer(self, model_name, lr, num_train_steps, num_warmup_steps, global_step=0):
        if model_name == 'bert':
            last_epoch = -1 if global_step == 0 else global_step
            param_optimizer = list(self.model.named_parameters())
            no_decay = ['bias', 'LayerNorm.weight']

            optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                 'weight_decay': 0.01},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]

            optimizer = AdamW(optimizer_grouped_parameters, lr=lr)
            for group in optimizer.param_groups:
                group['initial_lr'] = lr
            scheduler = get_linear_schedule_with_warmup(optimizer,
                                                        num_warmup_steps=num_warmup_steps,
                                                        num_training_steps=num_train_steps,
                                                        last_epoch=last_epoch)
        else:
            # for name, param in self.model.named_parameters():
            #     print(name, param.size())
            parameter_groups = []
            for name, param in self.model.named_parameters:
                if "bert" in name:
                    parameter_groups.append({
                        "params": param,
                        "lr": lr,
                    })
                else:
                    parameter_groups.append({
                        "params": param,
                        "lr": 1e-3,
                    })
            optimizer = AdamW(parameter_groups)
            # scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,
            #                                            mode='max',
            #                                            factor=0.5,
            #                                            patience=1,
            #                                            min_lr=0.0001,
            #                                            verbose=True)

        return optimizer, None

    def save_checkpoint(self, output_dir, acc):
        directory = output_dir + f'/{self.dataset_name}_checkpoint/'

        if not os.path.exists(directory):
            os.makedirs(directory)

        self.logger.info("saving model..")
        torch.save(self.model, directory + 'model.th')
        self.tokenizer.save_pretrained(directory)

    def train(self, train_data, epoch, eval_data, eval_epoch=1, eval=True):
        self.logger.info("***** training start *****")
        self.logger.info("Learning rate: " + f"{self.lr}")

        for epoch_id in range(epoch):
            print("Epoch:{}".format(epoch_id))

            # Run the train function
            pbar = tqdm(enumerate(train_data), total=len(train_data), ncols=50)
            # pbar = enumerate(data)
            for i, batch_data in pbar:
                loss_desc = self.train_batch(batch_data, self.clip)
                pbar.set_description(loss_desc)

            if eval and ((epoch_id + 1) % int(eval_epoch) == 0):
                eval_acc = self.evaluate(eval_data)
                # self.scheduler.step(eval_acc)
                if (eval_acc >= self.best_score):
                    self.best_score = eval_acc
                    self.save_checkpoint(args.model_dir, eval_acc)
                    print("MODEL SAVED")

    def train_batch(self, train_data, clip):
        self.model.train()
        self.train_loss = 0
        # Zero gradients of both optimizers
        self.optimizer.zero_grad()

        # Encode utterances
        output_state = self.model(train_data)
        labels = train_data["label"]

        # Loss calculation and backpropagation
        loss = CrossEntropyLoss()(output_state, labels)
        loss.backward()

        # Clip gradient norms
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip)

        # Update parameters with optimizers
        self.optimizer.step()

        self.train_loss += loss.item()
        train_loss_avg = self.train_loss / len(labels)
        # print('Loss:{}'.format(train_loss_avg))
        return 'Loss:{}'.format(train_loss_avg)

        # if i % 100 == 0 or i == len(data) - 1: # 每 100 epoch log一下
        #     self.logger.info(f"[Epoch {epoch+1} / Step {i}] " + "train_model " + "loss={:.4f}  train_acc={:.4f};".format(train_loss / n_data, train_acc))

    @torch.no_grad()
    def evaluate(self, eval_data):
        print("STARTING EVALUATION")
        self.model.eval()
        self.dev_loss = 0

        prediction = []
        total_label = []
        pbar = enumerate(eval_data)
        for i, batch_data in pbar:
            output_state = self.model(batch_data)
            loss = CrossEntropyLoss()(output_state, batch_data["label"])
            pred = torch.argmax(output_state, dim=1)
            prediction.extend(pred.cpu().numpy())
            total_label.extend(batch_data["label"].cpu().numpy())
            self.dev_loss += loss.item()

        dev_loss_avg = self.dev_loss / len(total_label)
        dev_acc = metrics.accuracy_score(total_label, prediction)
        print('Loss:{:.2f}'.format(dev_loss_avg), 'Acc:{:.2f}'.format(dev_acc))

        self.logger.info("Valid " + "loss={:.4f}  valid_acc={:.4f};".format(dev_loss_avg, dev_acc))
        return dev_acc
