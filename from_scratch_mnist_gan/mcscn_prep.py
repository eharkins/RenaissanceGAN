import subprocess
import os
out = open("logs/result.txt", 'w')
subprocess.run(["python3","mnist_gan.py", "100", "100"], stdout = out)
