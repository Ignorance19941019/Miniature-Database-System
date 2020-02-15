from BTree.btree import BTree
import numpy as np
import operator
from operator import itemgetter
import itertools 
import time
import re
import collections
database = {}
index_cache = {}
class Table(object):
    def __init__(self, file_path):
        self.file_path = file_path
    def createTable(self):
        lines = [line.strip('\n') for line in open(self.file_path)]
        records = [i.split('|') for i in lines]
        header = [i for i in records[0]]
        for i in range(len(records)):
            for j in range(len(header)):
                try:
                    records[i][j] = float(records[i][j])
                except ValueError:
                    continue  
        return records, header
    
class Index(object):
    def __init__(self, method):
        self.method = method
        
    def index(self, table, col):
        self.table = table
        header = [i for i in table[0]]
        if col not in header:
            print(col + " is not in the table.")
            return 
        col_index = header.index(col)
        self.hash_map = collections.defaultdict(set)
        for i in range(1, len(table)):
            self.hash_map[table[i][col_index]].add(i)
            self.mapping = self.hash_map
        if self.method == 'Btree':
            self.t = BTree()
            self.t.multi_insert(self.hash_map)         
            self.mapping = self.t
        return self.mapping
        
    def search(self, k):
        return self.mapping[k]  
    
def excute_query_(r, q):
    if 'inputfromfile' in q:
        func, arg = q.split('(')
        arg = arg.strip(')') + '.txt'
        database[r] = my_exec(func + "('" + arg + "'" +')')
    elif 'select' in q:
        left, right = q.split(',')
        func, table = left.split('(')
        table = "'" + table + "'"
        right = right.strip()[:-1]
        right = "'" + right + "'"
        database[r] = my_exec(func + "(" + table + ", " + right + ")") 
        outputtofile(r, r)
    elif 'join' in q:
        left, mid, right = q.split(',')
        func, table1 = left.split('(')
        func = func.strip()
        table1 = table1.strip()
        table1 = "'" + table1 + "'"
        table2 = mid.strip()
        table2 = "'" + table2 + "'"
        right = right.strip()[:-1]
        if '(' not in right:
            right = "(" + right + ")"
        right = "'" + right + "'"
        database[r] = my_exec(func + "(" + table1 + ", " + table2 + "," + right + ")") 
        outputtofile(r, r)
    else:
        func, args = q.split('(')
        args = args.strip(')')
        args = ",".join(["'" + arg.strip() + "'" for arg in args.split(',')])
        database[r] = my_exec(func + "( " + args + ")")
        outputtofile(r, r)
def excute_query(q):
    if 'outputtofile' in q:
        func, args = q.split('(')
        arg1, arg2 = args.split(',')
        arg1 = "'"+ arg1.strip() + "'"
        arg2 = arg2.strip(')')
        arg2 = "'" + arg2.strip() + "'"
        my_exec(func + "( " + arg1 + "," + arg2 +')')
    else:
        func, args = q.split('(')
        arg1, arg2 = args.split(',')
        full_attr = arg1.strip() + '_' + arg2.strip()
        arg1 = "'"+ arg1.strip() + "'"
        arg2 = arg2.strip(')')
        arg2 = "'" + arg2.strip() + "'"
        output = my_exec(func + "( " + arg1 + "," + arg2 +')')
        index_cache[full_attr] = output
    
    
def my_exec(code):
    exec('global i; i = %s' % code)
    global i
    return i

def inputfromfile(file_path):
    start_time = time.time()
    table = Table(file_path)
    end_time = time.time()
    print("Time elapsed for importing file is " + str(end_time - start_time) + " sec.")
    print("lines: " + str(len(table.createTable()[0])))
    return table.createTable()[0]

def column(matrix, i):
    return [row[i] for row in matrix]

def parse_select(arg):
    return re.sub('[\(\)\{\}]', '',arg)

def parse_join(arg):
    return re.sub('[\(\)\{\}]', '',arg)
    
def join(table_1, table_2, *argv):
    start_time = time.time()
    table1 = database[table_1]
    table2 = database[table_2]
    arithop = ['+', '-', '*', '/']
    header1 = [i for i in table1[0]]
    header2 = [i for i in table2[0]]
    new_header1 = [table_1 + '_' + str(i) for i in header1]
    new_header2 = [table_2 + '_' + str(i) for i in header2]
    argv = argv[0]
    if "and" in argv:
        args = [i.strip() for i in argv.split('and')]
    else:
        args = [argv.strip()]
    #first, pick out index of rows satisfying equality condition
    equal_args = [arg for arg in args if '=' in arg]
    idxes1 = set()
    idxes2 = set()
    equal_idxes = [set() for _ in range(len(equal_args))]
    for i, arg in enumerate(equal_args):
        condition = parse_join(arg)
        condition = condition.replace('=', '==')
        attr1 = [h for h in header1 if h in condition]
        if len(attr1) > 1:
            attr1 = [x for x in attr1 if x != table_2 and x != table_1]
        attr1_idx = header1.index(attr1[0])
        attr2 = [h for h in header2 if h in condition]
        if len(attr2) > 1:
            attr2 = [x for x in attr2 if x != table_1 and x != table_2]
        attr2_idx = header2.index(attr2[0])
        attr1 = attr1[0]
        attr2 = attr2[0]
        full_attr1 = table_1.strip() + '_' + attr1.strip()
        full_attr2 = table_2.strip() + '_' + attr2.strip()
        common = list(set(column(table2[1:], attr2_idx)) & set(column(table1[1:], attr1_idx))) 
        l = [[] for _ in range(len(common))]
        for n, element in enumerate(common):
            l1, l2 = [], []
            if len([a for a in condition if a in arithop]) == 0:
                if full_attr1 in index_cache.keys() and full_attr2 in index_cache.keys():
                    l1 = index_cache[full_attr1].search(element)
                    l2 = index_cache[full_attr2].search(element)
                elif full_attr1 in index_cache.keys():
                    l1 = index_cache[full_attr1].search(element)
                elif full_attr2 in index_cache.keys():
                    l2 = index_cache[full_attr2].search(element)
                if not l1:
                    for j in range(len(table1)):
                        if table1[j][attr1_idx] == element:
                            l1.append(j)
                if not l2:
                    for k in range(len(table2)):
                        if table2[k][attr2_idx] == element:
                            l2.append(k) 
                
            else:  
                for j in range(len(table1)):
                    if table1[j][attr1_idx] == element:
                        l1.append(j)
                for k in range(len(table2)):
                    if table2[k][attr2_idx] == element:
                        l2.append(k) 
            l[n] = list(itertools.product(l1,l2))
        l = list(itertools.chain(*l))
        equal_idxes[i].update(set(l))
    equal_idxes = set.intersection(*equal_idxes)
    t_keep = []
    t1_keep = []
    t2_keep = []
    for idx in equal_idxes:
        row1 = table1[idx[0]] 
        row2 = table2[idx[1]]
        t1_keep.append(row1)
        t2_keep.append(row2)
    # next part is to deal with non-equality join
    rest_args = [arg for arg in args if arg not in equal_args]
    if len(rest_args):
        rest_idxes = [set() for _ in range(len(rest_args))]
        for i, arg in enumerate(rest_args):
            condition = parse_join(arg)
            attr1 = [h for h in header1 if h in condition]
            if len(attr1) > 1:
                attr1 = [x for x in attr1 if x != table_2 and x != table_1]
            attr1_idx = header1.index(attr1[0])
            attr2 = [h for h in header2 if h in condition]
            if len(attr2) > 1:
                attr2 = [x for x in attr2 if x != table_1 and x != table_2]
            attr2_idx = header2.index(attr2[0])
            attr1 = attr1[0]
            attr2 = attr2[0]
            for j in range(len(t2_keep)):
                    condition_ = condition.replace(attr1, str(t1_keep[j][attr1_idx]))
                    condition_ = condition_.replace(table_1 +'.', '')
                    condition_ = condition_.replace(attr2, str(t2_keep[j][attr2_idx]))
                    condition_ = condition_.replace(table_2 + '.', '')
                    if eval(condition_):
                        rest_idxes[i].add(j)
            
        rest_idxes = set.intersection(*rest_idxes)
        for m in rest_idxes:
            row = t1_keep[m] + t2_keep[m]
            t_keep.append(row)
    else:
        for m in range(len(t1_keep)):
            row = t1_keep[m] + t2_keep[m]
            t_keep.append(row)
    
    new_header = list(new_header1 + new_header2) 
    end_time = time.time()
    print("Time elapsed for joining is " + str(end_time - start_time) + " sec.")
    print("lines: " + str(len([new_header] + t_keep)))
    return [new_header] + t_keep
  
def select(table_, *argv):
    start_time = time.time()
    table = database[table_]
    header = [i for i in table[0]]
    arithop = ['+', '-', '*', '/']
    argv = argv[0]
    if "and" in argv:
        args = [i.strip() for i in argv.split('and')]
    elif "or" in argv:
        args = [i.strip() for i in argv.split('or')]
    else:
        args = [argv.strip()]    
    idxes = [set() for _ in range(len(args))]
    for i, arg in enumerate(args):
        condition = parse_select(arg)
        condition = condition.replace('=', '==')
        attr = [h for h in header if h in condition][0]
        full_attr = table_.strip() + '_' + attr.strip()
        if "==" in condition and len([a for a in condition if a in arithop]) == 0 and full_attr in index_cache.keys():
            filter_value = int(condition.split('==')[-1].strip())
            idxes[i].update(index_cache[full_attr].search(filter_value))
        else:
            attr_idx = header.index(attr)
            for j in range(1, len(table)):
                    condition_ = condition.replace(attr, str(table[j][attr_idx]))
                    if eval(condition_):
                        idxes[i].add(j)
  
    if "and" in argv:
        idxes = set.intersection(*idxes)
    elif "or" in argv:
        idxes = set.union(*idxes)
    else:
        idxes = idxes[0]
    t_keep = []
    for idx in idxes:
        t_keep.append(table[idx])
    end_time = time.time()
    print("Time elapsed for selecting is " + str(end_time - start_time) + " sec.")
    print("lines: " + str(len([header] + t_keep)))
    return [header] + t_keep
        

def avg(table, col):
    start_time = time.time()
    res = []
    table = database[table]
    header = [i for i in table[0]]
    idx = header.index(col)
    avg = np.mean(column(table[1:], idx))
    end_time = time.time()
    print("The average of " + col + " is " + str(avg))
    print("Time elapsed for averaging is " + str(end_time - start_time) + " sec.")
    res.append(['avg_' + 'col'])
    res.append([avg])
    print("lines: " + str(len(res)))
    return res

def project(*argv):
    start_time = time.time()
    table = database[argv[0]]
    header = [i for i in table[0]]
    res = []
    for arg in argv[1:]:
        res.append(column(table, header.index(arg)))
    res = list(map(list, zip(*res)))
    end_time = time.time()
    print("Time elapsed for projection is " + str(end_time - start_time) + " sec.")
    print("lines: " + str(len(res)))
    return res

def sort(*argv):
    start_time = time.time()
    table = database[argv[0]]
    header = [i for i in table[0]]
    ids = [header.index(arg) for arg in argv[1:]]
    table[1:] = sorted(table[1:], key=operator.itemgetter(*ids)) 
    end_time = time.time()
    print("Time elapsed for sorting is " + str(end_time - start_time) + " sec.")
    print("lines: " + str(len(table)))
    return table

def sumgroup(table, col, *argv):
    table = database[table]
    start_time = time.time()
    header = [i for i in table[0]]
    res = []
    idxes = []
    col_index = header.index(col)
    for arg in argv:
        idxes.append(header.index(arg))
    grouper = itemgetter(*idxes)
    for key, grp in itertools.groupby(sorted(table[1:], key = grouper), grouper):
        if len(idxes) == 1:
            res.append([key, sum([int(v[col_index]) for v in grp])])
        else:
            res.append([*[k for k in key], sum([int(v[col_index]) for v in grp])])
    new_header =[]
    for arg in argv:
        new_header.append(arg)
    new_header.append('sum'+ "_"+ col)
    end_time = time.time()
    print("Time elapsed for sumgroup is " + str(end_time - start_time) + " sec.")
    print("lines: " + str(len([new_header] + res)))
    return [new_header] + res
    
def avggroup(table, col, *argv):
    table = database[table]
    start_time = time.time()
    header = [i for i in table[0]]
    res = []
    idxes = []
    col_index = header.index(col)
    for arg in argv:
        idxes.append(header.index(arg))
    grouper = itemgetter(*idxes)
    for key, grp in itertools.groupby(sorted(table[1:], key = grouper), grouper):
        if len(idxes) == 1:
            res.append([key, np.mean([int(v[col_index]) for v in grp])])
        else:
            res.append([*[k for k in key], sum([int(v[col_index]) for v in grp])])
            
    new_header =[]
    for arg in argv:
        new_header.append(arg)
    new_header.append('avg'+ "_"+ col)
    end_time = time.time()
    print("Time elapsed for avggroup is " + str(end_time - start_time) + " sec.")
    print("lines: " + str(len([new_header] + res)))
    return [new_header] + res

def countgroup(table, col, *argv):
    table = database[table]
    start_time = time.time()
    header = [i for i in table[0]]
    res = []
    idxes = []
    col_index = header.index(col)
    for arg in argv:
        idxes.append(header.index(arg))
    grouper = itemgetter(*idxes)
    for key, grp in itertools.groupby(sorted(table[1:], key = grouper), grouper):
        if len(idxes) == 1:
            res.append([key, len([int(v[col_index]) for v in grp])])
        else:
            res.append([*[k for k in key], sum([int(v[col_index]) for v in grp])])
    new_header =[]
    for arg in argv:
        new_header.append(arg)
    new_header.append('count'+ "_"+ col)
    end_time = time.time()
    print("Time elapsed for countgroup is " + str(end_time - start_time) + " sec.")
    print("lines: " + str(len([new_header] + res)))
    return [new_header] + res

def movsum(table, col, window_size_):
    window_size = int(window_size_)
    table = database[table]
    start_time = time.time()
    header = [i for i in table[0]]
    col_index = header.index(col)
    col_ = column(table, col_index)[1:]
    res = [sum(col_[i-(window_size-1):i+1]) if i>(window_size-1) else sum(col_[:i+1])  for i in range(len(col_))]
    header += [col + "_sum"]
    ans = []
    ans.append(header)
    for i in range(1, len(table)):
        temp = table[i] + [res[i - 1]]
        ans.append(temp)
    end_time = time.time()
    print("Time elapsed for moving sum is " + str(end_time - start_time) + " sec.")
    print("lines: " + str(len(ans)))
    return ans
    

def movavg(table_, col, window_size_):
    window_size = int(window_size_)
    table = database[table_]
    start_time = time.time()
    header = [i for i in table[0]]
    header += [col + "_avg"]
    res = column(movsum(table_, col, window_size_)[1:], -1)
    for i in range(window_size):
        res[i] /= i + 1
    for i in range(window_size, len(res)):
        res[i] /= window_size
    ans = []
    ans.append(header)
    for i in range(1, len(table)):
        temp = table[i] + [res[i - 1]]
        ans.append(temp)
    end_time = time.time()
    print("Time elapsed for moving avg is " + str(end_time - start_time) + " sec.")
    print("lines: " + str(len(ans)))
    return ans

def Btree(table, col):
    table = database[table]
    mapping = Index('Btree')
    Indexed = mapping.index(table, col)
    return mapping  
  
def Hash(table, col):
    table = database[table]
    mapping = Index('Hash')
    Indexed = mapping.index(table, col)
    return mapping  
    

def concat(table1, table2):
    table1 = database[table1]
    table2 = database[table2]
    start_time = time.time()
    header1 = [i for i in table1[0]]
    header2 = [i for i in table2[0]]
    if set(header1) != set(header2):
        print("Error: two tables don't share the same schema!")
        return 
    end_time = time.time()
    print("Time elapsed for concatenation is " + str(end_time - start_time) + " sec.")
    print("lines: " + str(len(table1 + table2[1:])))
    return table1 + table2[1:]

def outputtofile(table, file):
    table = database[table]
    with open('nl1668_lz1883_' + file + '.txt', 'w') as f:
        for list_ in table:
            f.write("|".join([str(list_[i]) for i in range(len(list_))]) + '\n')
    
