from pathlib import Path


        
txt = Path("C:/Users/cmatz/master-thesis/fair-and-private-lm/datasets/wikipedia_bookcorpus/original-data.txt").resolve()

# read number of rows quickly
length = sum(1 for row in open(txt, 'r',  encoding="utf8"))
print(length)

#141598553
#255261105