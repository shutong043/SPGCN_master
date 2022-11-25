
import os
for j in range(5):
    # 选择j=0和j=1来复现SPGCN-9和SPGCN-21
    for i in range(1):
        os.system("python ./IP_pre.py %s" % (i))
        os.system("python ./IP_final.py %s %s" % (i, j))
        os.system("python ./IP_test.py %s %s" % (i, j))
for j in range(5):
    #选择j=0和j=1来复现SPGCN-9和SPGCN-21
    for i in range(1):
        os.system("python ./PU_pre.py %s" % (i))
        os.system("python ./PU_final.py %s %s" % (i, j))
        os.system("python ./PU_test.py %s %s " % (i, j))
for j in range(5):
    #选择j=0和j=1来复现SPGCN-9和SPGCN-21
    for i in range(1):
        os.system("python ./Sa_pre.py %s" % (i))
        os.system("python ./Sa_final.py %s %s" % (i, j))
        os.system("python ./Sa_test.py %s %s" % (i, j))
for j in range(5):
    #选择j=0和j=3来复现SPGCN-9和SPGCN-21
    for i in range(1):
        os.system("python ./Ht_pre.py %s" % (i))
        os.system("python ./Ht_final.py %s %s" % (i, j))
        os.system("python ./Ht_test.py %s %s" % (i, j))
for j in range(5):
    #选择j=0和j=1来复现SPGCN-9和SPGCN-21
    for i in range(1):
        os.system("python ./PU_pre.py %s" % (i))
        os.system("python ./PU_final.py %s %s" % (i, j))
        os.system("python ./PU_test.py %s %s " % (i, j))