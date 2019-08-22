import psutil
import numpy as np
from time import sleep

x = []
for i in range(60):
    x.append(psutil.cpu_percent())
    sleep(1)
    
print(np.mean(x) / 100. * 45)
