# Project: A miniature relational database with order

## 
This the course project for DSGA 2433 - Database Systems at NYU.

### Prerequisites

Dependencies: numpy, sys, time, itertools, operator, re and BTree implementation. The BTree implementation is taken from https://github.com/SimonCqk/BTree. 

### Running the test

This program takes input_path of tests.  Run this program using 
```
python3 main.py input_path

```
My own test file is called test.txt. Run using
```
python3 main.py test.txt

```

The program will print the running time for each single query and produce a text file of the result table (if the input query generates one).  This program assumes 'sales1.txt' and 'sales2.txt' to be in the same directory as test.txt, util.py and main.py. 


