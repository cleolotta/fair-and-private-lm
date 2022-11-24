from pathlib import Path

txt = Path("C:/Users/cmatz/master-thesis/fair-and-private-lm/data_prep/data.txt").resolve()
txt2 = Path("C:/Users/cmatz/master-thesis/fair-and-private-lm/data_prep/original-test.txt").resolve()
txt_b = Path("C:/Users/cmatz/master-thesis/fair-and-private-lm/data_prep/original-train.txt").resolve()
#txt_c = Path("C:/Users/cmatz/master-thesis/fair-and-private-lm/data-prep/datasets/original-wiki-test.txt").resolve()



length = sum(1 for row in open(txt, "r", encoding= "utf-8"))
length_a = sum(1 for row in open(txt2, "r", encoding= "utf-8"))
length_b = sum(1 for row in open(txt_b, "r", encoding= "utf-8"))
#length_c = sum(1 for row in open(txt_c, "r", encoding= "utf-8"))

#txt2 = Path("/ukp-storage-1/matzken/fair-and-private-lm/original_data_mlm/orig_output_train.txt").resolve()

#length2 =sum(1 for row in open(txt2, "r", encoding= "utf-8"))

#txt3 = Path("/ukp-storage-1/matzken/fair-and-private-lm/original_data_mlm/orig_output_test.txt").resolve()

#length3 = sum(1 for row in open(txt3, "r", encoding= "utf-8"))

#print(f"original: {length}, train: {length2}, test: {length3}")
#print("---------------------------------------------------------")
#print(f"train+test = {length2 + length3}, original = {length}, original -train -test = {length - length2 - length3}")
print(f"size of dataset: {length}")
print(f"test: {length_a}")
print(f"train: {length_b}")


