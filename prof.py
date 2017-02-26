#import cProfile as prof


import test_simanneal


OUTPATH = "./outputs/015"

class FakeArgs(object):
    scenario = "trending_today"


args = FakeArgs()

test_simanneal.STEPS = 10
exec("test_simanneal.main(args, OUTPATH)")
