import subprocess
import os
out = open("logs/result.txt", 'w')
# module load slurm
#   module load python-3.6.3
#   module load opencv-3.3.0
#   srun —-pty bash -i
#   cd SeniorProject/from_scratch_mnist_gan
# subprocess.run(["module"])
crop = "module"
subprocess.call(crop, shell=True)

# subprocess.run(["module load", "python-3.6.3"])
# subprocess.run(["module load", "opencv-3.3.0"])
# subprocess.run(["srun", "--pty", "bash", "-i"])
# subprocess.run(["python3","mnist_gan.py", "100", "100"], stdout = out)
