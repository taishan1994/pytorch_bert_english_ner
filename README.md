# pytorch_bert_english_ner
基于bert的英文实体识别。

英文和中文不同，英文之间的单词是由空格隔开的。bert对英文进行编码时会删除掉单词之间的空格，而且会将单词拆分为子词，因此需要对标签进行重整。

# 依赖

```python
transformers==4.7.0
```

# 运行

这里使用的是conll2003数据集，并稍作处理，具体见data/conll2003/mid_data下的数据，一般格式为：

```python
{"id": 0, "text": ["SOCCER", "-", "JAPAN", "GET", "LUCKY", "WIN", ",", "CHINA", "IN", "SURPRISE", "DEFEAT", "."], "labels": ["O", "O", "B-LOC", "O", "O", "O", "O", "B-PER", "O", "O", "O", "O"]}
```

训练、验证、测试和预测都在main.py中。

```python
python main.py
```

注意：在验证和测试的时候是只针对于实体及其类别，没有考虑到其位置。

```python
          precision    recall  f1-score   support

     PER       0.96      0.96      0.96      1612
     LOC       0.93      0.92      0.93      1643
    MISC       0.79      0.77      0.78       690
     ORG       0.90      0.88      0.89      1636

micro-f1       0.91      0.90      0.91      5581


example = {"id": 13, "text": ["Nader", "Jokhadar", "had", "given", "Syria", "the", "lead", "with", "a", "well-struck", "header", "in", "the", "seventh", "minute", "."], "labels": ["B-PER", "I-PER", "O", "O", "B-LOC", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O"]}
{'PER': [['Na', '##der', 'Jo', '##kha', '##dar']], 'LOC': [['Syria']], 'MISC': [], 'ORG': []}
```

最后需要将拆分的实体进行合并，这里没有实现，可自行实现。如果还需要实体的位置，可通过正则进行匹配。

# 补充

Q：如何训练自己的数据？

A：参考conll2003数据集即可。
