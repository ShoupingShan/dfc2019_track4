import os
import sys
BASE_DIR = os.argv[0]
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
print(BASE_DIR)
print(sys.path)
