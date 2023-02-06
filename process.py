import re
import json

all_labels = set()

def process_conll2003(in_path, out_path):
  with open(in_path, "r") as fp:
    data = fp.read()
  data = re.split("\n\n", data)
  fp = open(out_path, "w", encoding="utf-8")
  for i in range(len(data)):
    if i == 0:
      continue
    details = data[i].split("\n")
    text = []
    labels = []
    output = {}
    output["id"] = i - 1
    for detail in details:
      tmp = detail.split(" ")
      word = tmp[0]
      label = tmp[-1]
      if "-" in label:
        rel_label = label.split("-")
        if len(rel_label) == 2:
          all_labels.add(rel_label[-1])
      text.append(word)
      labels.append(label)
    output["text"] = text
    output["labels"] = labels
    if text == ["-DOCSTART-"]:
      continue
    assert len(text) == len(labels)
    fp.write(json.dumps(output) + "\n")
  fp.close()

process_conll2003("data/conll2003/raw_data/train.txt", "data/conll2003/mid_data/train.txt")
process_conll2003("data/conll2003/raw_data/valid.txt", "data/conll2003/mid_data/valid.txt")
process_conll2003("data/conll2003/raw_data/test.txt", "data/conll2003/mid_data/test.txt")

with open("data/conll2003/mid_data/labels.txt", "w") as fp:
  fp.write("\n".join(list(all_labels)))