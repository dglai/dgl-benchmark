Relational-GCN
==============

Entity Classification
---------------------
AIFB:
```bash
python3 entity_classify.py -d aifb --testing --gpu 0
```

MUTAG:
```bash
python3 entity_classify.py -d mutag --l2norm 5e-4 --n-bases 30 --testing --gpu 0
```

BGS:
```bash
python3 entity_classify.py -d bgs --l2norm 5e-4 --n-bases 40 --testing --gpu 0 --relabel
```
