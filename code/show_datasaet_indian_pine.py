import os
for dirname, _, filenames in os.walk('C:\\Users\\AssaneThiao\\Documents\\internship\\dataset'):
    for filename in filenames:
        print(os.path.join(dirname, filename))