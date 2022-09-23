from pathlib import Path

txt = Path("C:/Users/cmatz/master-thesis/fair-and-private-lm/data-prep/datasets/augmented_data_big.txt").resolve()
txt2 = Path("C:/Users/cmatz/master-thesis/fair-and-private-lm/data-prep/datasets/original_data_big.txt").resolve()

#txt2 = Path("C:/Users/cmatz/master-thesis/fair-and-private-lm/data-prep/datasets/original-test1.txt").resolve()
#txt_a = Path("C:/Users/cmatz/master-thesis/fair-and-private-lm/data-prep/datasets/augmented-train1.txt").resolve()
#txt_a2 = Path("C:/Users/cmatz/master-thesis/fair-and-private-lm/data-prep/datasets/augmented-test1.txt").resolve()



length = sum(1 for row in open(txt, "r", encoding= "utf-8"))
length2 = sum(1 for row in open(txt2, "r", encoding= "utf-8"))
#length_a = sum(1 for row in open(txt_a, "r", encoding= "utf-8"))
#lenght_a2 = sum(1 for row in open(txt_a2, "r", encoding= "utf-8"))

#print("augmented train ", length_a)
#print("augmented test ", lenght_a2)
print("augmented big ", length)
print("original big ", length2)

