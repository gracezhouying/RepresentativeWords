# RepresentativeWords
Text Mining - Find most Representative Words in a list. 
Use this list to classify text files and compare result with random selection.

## Vocabulary selection:
Goal: Given full vocabulary list and number M, find the most representative M words from full list.

First, classify training text documents into 20 categories. 
Use list named classes to store the 20 categories documents. 
Create a 20*len(full vocabulary) matrix to store the counts of vocabularies in different classes. 
The ijth element of this matrix records the occurrence times of full vocabulary’s jth word in class i. 
Calculate the mean and max value of each column of this matrix. 
Define difference = max – mean, find the indexes of the M biggest differences. 
A large difference value indicates the corresponding vocabulary is representative of a specific class. 
Thus I will choose the most representative M vocabularies.

## Document Classification Algorithm:
Goal: given selected vocabulary list V, classify test document x.

Create a 20*len(V) Matrix, the ijth element records the occurrence times of V’s jth word in class i. 
Create a list x_occur of size len(V) to record the occurrence times of V’s words in document x.
For document x, find the class j that maximizes 𝑙𝑜𝑔𝜋𝑗 + ∑|𝑉| 𝑥_𝑜𝑐𝑐𝑢𝑟𝑖 𝑙𝑜𝑔𝑝𝑗𝑖 , 𝑖=1
here, 𝑝𝑗𝑖 = (0.1 + 𝑀𝑎𝑡𝑟𝑖𝑥[𝑗][𝑖])/(𝑀𝑎𝑡𝑟𝑖𝑥[𝑗]. 𝑠𝑢𝑚 + 0.1 ∗ 𝑙𝑒𝑛(𝑉) ). 0.1 is smoothing.
