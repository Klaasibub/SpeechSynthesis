import os



files = sorted([int(file.split('_')[-1]) for file in os.listdir('/storage/sidenko/voco_checkpoints') if len(file.split('_')) > 1])
for file in files[:-3]:
    os.remove(f'/storage/sidenko/voco_checkpoints/waveglow_{file}')
