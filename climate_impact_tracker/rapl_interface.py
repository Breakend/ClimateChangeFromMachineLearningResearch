from ctypes import *
import ctypes as c
class RaplTollInterface(object):

    def __init__(self):
        self.lib = cdll.LoadLibrary("vector_python_lib.so")
        # self.lib.GetPowerStats.restype = c_double
        # self.lib.GetPowerStats.argtypes = [c_int, c_int]

        prototype = c.CFUNCTYPE(    
            c.c_double,                
            c.c_int,                
            c.c_int                
        )
        self.GetPowerStats = prototype(('GetPowerStats', self.lib))

    def get_stats(self):
        return self.GetPowerStats(c.c_int(100), c.c_int(10))

if __name__ == "__main__":
    face = RaplTollInterface()
    face.get_stats()