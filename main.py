from util import excute_query_, excute_query
from util import database
import sys
def main(test_file):
    queries = [query.rstrip('\n') for query in open(test_file)]
    valid_queries = []
    for query in queries:
        if query.split()[0].strip() != "//":
            valid_queries.append(query.split("//")[0].strip())
    for query in valid_queries:
        if ":=" in query:
            r, q = query.split(":=")
            r = r.strip()
            q = q.strip()
            excute_query_(r, q)

        else:
            excute_query(query)
    l = []
    for table in database:
        l.append(database[table])
    with open('nl1668_lz1883_AllOperations.txt', 'w') as f:
        for table in l:
            f.write('-------------------------------------------' + '\n')
            for list_ in table:
                f.write("|".join([str(list_[i]) for i in range(len(list_))]) + '\n')
        
    
        
if __name__ == '__main__':
    testfile = sys.argv[1]
    main(testfile) 
    
