import os


files = sorted([int(file.split('_')[-1]) for file in os.listdir('/home/sidenko/my/output') if file.startswith('check') and len(file.split('_')) > 1])
for file in files[:-3]:
    os.remove(f'/home/sidenko/my/output/checkpoint_{file}')
