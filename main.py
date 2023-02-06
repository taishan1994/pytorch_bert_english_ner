import json
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup, BertModel, BertConfig

from metrics import calculate_metric, get_p_r_f, classification_report

def conver_labels_to_biolabels(labels):
    label2id = {"O": 0}
    id2label = {0: "O"}
    i = 1
    for label in labels:
        tmp_label = "B-" + label
        label2id[tmp_label] = i
        id2label[i] = tmp_label
        i += 1
        tmp_label = "I-" + label
        label2id[tmp_label] = i
        id2label[i] = tmp_label
        i += 1
    return label2id, id2label

def bio_decode(text, label, id2label, ent_labels):
    entities = {label: [] for label in ent_labels}
    label = [id2label[lab] if lab != -100 else "O" for lab in label]
    for i,(tex, lab) in enumerate(zip(text, label)):
      if "B-" in lab:
        tmp = []
        tmp.append(tex)
        elabel = lab.split("-")[-1]
        i += 1
        while i < len(text) and "I-" + elabel == label[i]:
            tmp.append(text[i])
            i += 1
        if tmp not in entities[elabel]:
            entities[elabel].append(tmp)
    return entities
      



class NerDataset(Dataset):
    def __init__(self, examples, args):
        self.examples = examples
        self.tokenizer = args.tokenizer
        self.max_seq_len = args.max_seq_len
        self.label2id = args.label2id

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        example = self.examples[item]
        example = json.loads(example)
        text = example["text"]
        labels = example["labels"]
        text2 = self.tokenizer.tokenize(" ".join(text))
        # print(text)
        # print(text2)
        # print(labels)
        token_type_ids = [0] * self.max_seq_len
        label_ids = self.align_label_example(text, text2, labels)
        # print(label_ids)
        if len(text2) > self.max_seq_len - 2:
            attention_mask = [1] * self.max_seq_len
            input_ids = self.tokenizer.convert_tokens_to_ids(['[CLS]'] + text2[:self.max_seq_len - 2] + ['[SEP]'])
            label_ids = [-100] + label_ids[:self.max_seq_len - 2] + [-100]
        else:
            attention_mask = [1] * (len(label_ids) + 2) + [0] * (self.max_seq_len - len(label_ids) - 2)
            input_ids = self.tokenizer.convert_tokens_to_ids(['[CLS]'] + text2 + ['[SEP]']) + [0] * (
                    self.max_seq_len - len(text2) - 2)
            label_ids = [-100] + label_ids + [-100] + [-100] * (self.max_seq_len - 2 - len(label_ids))

        assert len(input_ids) == self.max_seq_len
        assert len(attention_mask) == self.max_seq_len
        assert len(token_type_ids) == self.max_seq_len
        assert len(label_ids) == self.max_seq_len

        input_ids = torch.tensor(np.array(input_ids)).long()
        attention_mask = torch.tensor(np.array(attention_mask)).long()
        token_type_ids = torch.tensor(np.array(token_type_ids)).long()
        label_ids = torch.tensor(np.array(label_ids)).long()

        output = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "labels": label_ids,
        }
        return output

    def align_label_example(self, ori_input, tokenized_input, label, label_all_tokens=True):
        """这里目前label_all_tokens只能设置为True"""
        i, j = 0, 0
        ids = []
        # 这里需要遍历tokenizer后的列表
        while i < len(tokenized_input):
            if tokenized_input[i] == ori_input[j]:
                ids.append(self.label2id[label[j]])
                # ids.append(label[j])
                i += 1
                j += 1
            else:
                tmp = []
                tmp.append(tokenized_input[i])  # 将当前的加入的tmp
                ids.append(self.label2id[label[j]])  # 当前的id加入到ids
                i += 1
                while i < len(tokenized_input) and "".join(tmp) != ori_input[j]:
                    ori_word = tokenized_input[i]
                    if ori_word[:2] == "##":
                        tmp.append(ori_word[2:])
                        if label[j] == "O":
                          ids.append(self.label2id[label[j]])
                        else:
                          ids.append(self.label2id["I-" + label[j].split("-")[-1]] if label_all_tokens else -100)
                    else:
                        if label[j] == "O":
                            ids.append(self.label2id[label[j]])
                        else:
                            if "O" == label[j]:
                                ids.append(self.label2id[label[j]])
                            else:
                                ids.append(self.label2id["I-" + label[j].split("-")[-1]])
                        tmp.append(ori_word)
                        # ids.append(label[j])
                    i += 1
                j += 1
        assert len(ids) == len(tokenized_input)
        return ids

class NerModel(nn.Module):
    def __init__(self, args):
        super(NerModel, self).__init__()
        self.args = args
        self.bert = BertModel.from_pretrained(args.bert_dir)
        self.config = BertConfig.from_pretrained(args.bert_dir)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.config.hidden_size, args.num_labels)
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(self,
          input_ids,
          token_type_ids,
          attention_mask,
          labels):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss = self.criterion(logits.view(-1, self.args.num_labels), labels.view(-1))

        return loss, logits


class NerPipeline:
    def __init__(self, model, args):
        self.model = model
        self.args = args

    def save_model(self):
        torch.save(self.model.state_dict(), self.args.save_dir)

    def load_model(self):
        self.model.load_state_dict(torch.load(self.args.save_dir, map_location="cpu"))

    def build_optimizer_and_scheduler(self, t_total):
        module = (
            self.model.module if hasattr(self.model, "module") else self.model
        )

        # 差分学习率
        no_decay = ["bias", "LayerNorm.weight"]
        model_param = list(module.named_parameters())

        bert_param_optimizer = []
        other_param_optimizer = []

        for name, para in model_param:
            space = name.split('.')
            # print(name)
            if "bert" in space[0]:
                bert_param_optimizer.append((name, para))
            else:
                other_param_optimizer.append((name, para))

        optimizer_grouped_parameters = [
            # bert other module
            {"params": [p for n, p in bert_param_optimizer if not any(nd in n for nd in no_decay)],
             "weight_decay": self.args.weight_decay, 'lr': self.args.lr},
            {"params": [p for n, p in bert_param_optimizer if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0, 'lr': self.args.lr},

            # 其他模块，差分学习率
            {"params": [p for n, p in other_param_optimizer if not any(nd in n for nd in no_decay)],
             "weight_decay": self.args.weight_decay, 'lr': self.args.other_lr},
            {"params": [p for n, p in other_param_optimizer if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0, 'lr': self.args.other_lr},
        ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.lr, eps=self.args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=int(self.args.warmup_proportion * t_total), num_training_steps=t_total
        )

        return optimizer, scheduler

    def eval_forward(self, data_loader):
        span_logits = None
        span_labels = None
        self.model.eval()
        for eval_step, batch_data in enumerate(data_loader):
            for key in batch_data.keys():
                batch_data[key] = batch_data[key].to(self.args.device)
            labels = batch_data["labels"]
            output = self.model(batch_data["input_ids"],
                      batch_data["token_type_ids"],
                      batch_data["attention_mask"],
                      labels)
            labels = labels.detach().cpu().numpy()
            loss, span_logit = output
            span_logit = span_logit.detach().cpu().numpy()
            span_logit = np.argmax(span_logit, -1)
            if span_logits is None:
                span_logits = span_logit
                span_labels = labels
            else:
                span_logits = np.append(span_logits, span_logit, axis=0)
                span_labels = np.append(span_labels, labels, axis=0)

        return span_logits, span_labels

    def get_metric(self, span_logits, span_labels, callback):
        batch_size = len(callback)
        total_count = [0 for _ in range(len(self.args.labels))]
        role_metric = np.zeros([len(self.args.labels), 3])
        for span_logit, label, tokens in zip(span_logits, span_labels, callback):
            span_logit = span_logit[1:len(tokens)+1]
            label = label[1:len(tokens)+1]
            pred_entities = bio_decode(tokens, span_logit, self.args.id2label, self.args.labels)
            gt_entities = bio_decode(tokens, label, self.args.id2label, self.args.labels)
            # print("========================")
            # print(tokens)
            # print(label)
            # print(span_logit)
            # print(pred_entities)
            # print(gt_entities)
            # print("========================")
            for idx, _type in enumerate(self.args.labels):
                if _type not in pred_entities:
                    pred_entities[_type] = []
                if _type not in gt_entities:
                    gt_entities[_type] = []
                total_count[idx] += len(gt_entities[_type])
                role_metric[idx] += calculate_metric(pred_entities[_type], gt_entities[_type])

        return role_metric, total_count

    def train(self, train_loader, dev_loader=None):

        t_total = len(train_loader) * self.args.train_epoch
        optimizer, scheduler = self.build_optimizer_and_scheduler(t_total)

        global_step = 0
        self.model.zero_grad()
        self.model.to(self.args.device)
        eval_step = self.args.eval_step
        best_f1 = 0.
        for epoch in range(1, self.args.train_epoch + 1):
            for step, batch_data in enumerate(train_loader):
                self.model.train()
                for key in batch_data.keys():
                    batch_data[key] = batch_data[key].to(self.args.device)
                labels = batch_data["labels"]
                output = self.model(batch_data["input_ids"],
                                    batch_data["token_type_ids"],
                                    batch_data["attention_mask"],
                                    labels)
                loss, logits = output
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                loss.backward()
                optimizer.step()
                scheduler.step()
                self.model.zero_grad()
                global_step += 1
                print('【train】Epoch: %d/%d Step: %d/%d loss: %.5f' % (
                    epoch, self.args.train_epoch, global_step, t_total, loss.item()))
                if dev_loader is not None and global_step % eval_step == 0:
                  span_logits, span_labels = self.eval_forward(dev_loader)
                  role_metric, total_count = self.get_metric(span_logits, span_labels, dev_callback)
                  mirco_metrics = np.sum(role_metric, axis=0)
                  mirco_metrics = get_p_r_f(mirco_metrics[0], mirco_metrics[1], mirco_metrics[2])
                  p, r, f = mirco_metrics[0], mirco_metrics[1], mirco_metrics[2]
                  print('【eval】 precision={:.4f} recall={:.4f} f1_score={:.4f}'.format(mirco_metrics[0], mirco_metrics[1], mirco_metrics[2]))
                  if f > best_f1:
                    best_f1 = f
                    print("【best f1】 {:.4f}".format(mirco_metrics[2]))
                    self.save_model()
        if dev_loader is None:
          self.save_model()

    def test(self, test_loader):
        self.load_model()
        self.model.to(self.args.device)
        with torch.no_grad():
          span_logits, span_labels = self.eval_forward(test_loader)
          role_metric, total_count = self.get_metric(span_logits, span_labels, test_callback)
          print(self.args.labels)
          print(classification_report(role_metric, self.args.labels, {i:label for i,label in enumerate(self.args.labels)}, total_count))

    def predict(self, text):
        self.load_model()
        self.model.eval()
        self.model.to(self.args.device)
        with torch.no_grad():
            tmp_tokens = self.args.tokenizer.tokenize(text)
            encode_dict = self.args.tokenizer.encode_plus(text=tmp_tokens,
                                    max_length=self.args.max_seq_len,
                                    padding="max_length",
                                    truncating="only_first",
                                    return_token_type_ids=True,
                                    return_attention_mask=True)
            tokens = ['[CLS]'] + tmp_tokens + ['[SEP]']
            token_ids = torch.from_numpy(np.array(encode_dict['input_ids'])).unsqueeze(0).to(self.args.device)
            attention_mask = torch.from_numpy(np.array(encode_dict['attention_mask'])).unsqueeze(0).to(
                self.args.device)
            token_type_ids = torch.from_numpy(np.array(encode_dict['token_type_ids'])).unsqueeze(0).to(self.args.device)
            output = self.model(token_ids, token_type_ids, attention_mask, labels=None)
            loss, logit = output
            logit = logit.detach().cpu().numpy()
            logit = np.argmax(logit, -1)
            logit = logit[0][1:len(tokens)+1]
            print(bio_decode(tmp_tokens, logit, self.args.id2label, self.args.labels))


if __name__ == "__main__":
    class Args:
        data_name = "conll2003"
        save_dir = "checkpoints/{}/model.pt".format(data_name)
        bert_dir = "bert-base-cased"
        with open("data/conll2003/mid_data/labels.txt", "r") as fp:
            labels = fp.read().strip().split("\n")
        label2id, id2label = conver_labels_to_biolabels(labels)
        print(label2id)
        print(id2label)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = BertTokenizer.from_pretrained(bert_dir, do_lower_case=False)

        train_epoch = 3
        max_seq_len = 128
        train_batch_size = 64
        eval_batch_size = 64
        eval_step = 100
        lr = 3e-5
        other_lr = 2e-3
        adam_epsilon = 1e-8
        warmup_proportion = 0.1
        max_grad_norm = 5
        weight_decay = 0.01
        num_labels = len(labels) * 2 + 1


    args = Args()

    with open("data/conll2003/mid_data/train.txt", "r") as fp:
        train_examples = fp.read().strip().split("\n")

    train_dataset = NerDataset(train_examples, args)

    with open("data/conll2003/mid_data/valid.txt", "r") as fp:
        dev_examples = fp.read().strip().split("\n")
    
    dev_callback = [args.tokenizer.tokenize(" ".join(json.loads(example)["text"]))[:args.max_seq_len-2] for example in dev_examples]
    dev_dataset = NerDataset(dev_examples, args)
 
    with open("data/conll2003/mid_data/test.txt", "r") as fp:
        test_examples = fp.read().strip().split("\n")
    print(test_examples[0])
    test_callback = [args.tokenizer.tokenize(" ".join(json.loads(example)["text"]))[:args.max_seq_len-2] for example in test_examples]
    test_dataset = NerDataset(test_examples, args)


    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.train_batch_size, num_workers=1)
    dev_loader = DataLoader(dev_dataset, shuffle=False, batch_size=args.eval_batch_size, num_workers=1)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=args.eval_batch_size, num_workers=1)

    # for i, data in enumerate(train_loader):
    #     for k, v in data.items():
    #         print(k, v)
    #     break

    model = NerModel(args)

    nerPipeline = NerPipeline(model, args)
    nerPipeline.train(train_loader, dev_loader=dev_loader)
    nerPipeline.test(test_loader)

    example = {"id": 13, "text": ["Nader", "Jokhadar", "had", "given", "Syria", "the", "lead", "with", "a", "well-struck", "header", "in", "the", "seventh", "minute", "."], "labels": ["B-PER", "I-PER", "O", "O", "B-LOC", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O"]}
    text = example["text"]
    text = " ".join(text)
    print(text)
    nerPipeline.predict(text)

