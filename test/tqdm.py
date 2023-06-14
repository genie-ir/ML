import glob, os
r = glob.glob(os.path.join('/home/alihejrati/Desktop/untitled folder', "**", f"*.txt"))
r = sorted([os.path.relpath(p, start='/home/alihejrati/Desktop/untitled folder') for p in r])
print(r)
from time import sleep
from tqdm import tqdm
for ir in tqdm(r, leave=False):
    sleep(3)
    print(ir)