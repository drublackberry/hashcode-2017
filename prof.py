#import cProfile as prof


import test_simanneal


OUTPATH = "./outputs/088"

class FakeArgs(object):
    scenario = "videos_worth_spreading"


args = FakeArgs()

test_simanneal.STEPS = 10
exec("test_simanneal.main(args, OUTPATH)")
