import iolib
import rules
import model
import time
import scipy.sparse as sp

FILE = "./outputs/042/kittens-00001.out"
#FILE = "./outputs/001/me_at_the_zoo-00001.out"

#mod = model.SparseModel("me_at_the_zoo")
mod = model.SparseModel("kittens")
S = iolib.read_output(FILE, mod.C, mod.V)

judge = rules.Judge(mod)

print("START")
t0 = time.time()
for i in range(1):
    sc = judge.score(S, ignore_overflow=False)
t1 = time.time()
print((t1 - t0) / 1)

print(sc)
