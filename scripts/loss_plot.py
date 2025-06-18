import re
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np

files = ['to200.out','to230.out','to530.out','to830.out']
epochs = []
losses = []
val_epochs = []
class_ap = defaultdict(list)
class_names = []
ap_keys = ['AP', 'AP_50', 'AP_25']
epoch_offset = 0
epoch_times = []
for file in files:
    
    with open(f'data/logs/{file}', 'r') as f:
        lines = f.readlines()

    

    i = 0
    while i < len(lines):


        

            
            


        line = lines[i]
        if 'Checkpoint created' in line:
                match = re.search(r'\[(\d+):(\d+)<', line)
                if match:
                    minutes = int(match.group(1))
                    seconds = int(match.group(2))
                    total_seconds = minutes * 60 + seconds
                    epoch_times.append(total_seconds)
        epoch_match = re.search(r'Epoch (\d+)', line)
        loss_match = re.search(r'loss=(\d+)', line)
        if epoch_match and loss_match:
            if int(loss_match.group(1))<20:
                i +=1
                continue
            epochs.append(epoch_offset + int(epoch_match.group(1)))
            losses.append(int(loss_match.group(1)))
        
        if line.startswith('###############################') and i + 7 < len(lines) and i>170:
            block_lines = lines[i+3:i+9]
            for class_line in block_lines:
                parts = class_line.strip().split()
                if len(parts) == 5:
                    name = parts[0]
                    if name not in class_names:
                        class_names.append(name)
                    ap_values = list(map(float, parts[2:]))
                    for k, val in zip(ap_keys, ap_values):
                        class_ap[(name, k)].append(val)
            val_epochs.append(epochs[-1] if epochs else 0)      
            i += 7
        i += 1
    epoch_offset = epochs[-1] if epochs else 0


epoch_times = np.array(epoch_times)
mean_time = np.mean(epoch_times)
std_time = np.std(epoch_times)

print(f'Average time per epoch: {mean_time:.2f} s')
print(f'Standard deviation: {std_time:.2f} s')

plt.plot(epochs, losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True)
plt.show()

class_name_translation = {'wall':'cylinder','board':'ladder','column':'cable','clutter':'noise','average':'average'}


for name in class_names:
    plt.plot(val_epochs, class_ap[(name, 'AP')], label=class_name_translation[name])

plt.xlabel('Epoch')
plt.ylabel('AP')
plt.legend(fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()



fig, axs = plt.subplots(len(class_names), 1, figsize=(8, 3 * len(class_names)), sharex=True)
for idx, name in enumerate(class_names):
    axs[idx].plot(val_epochs, class_ap[(name, 'AP')], label='AP')
    axs[idx].plot(val_epochs, class_ap[(name, 'AP_50')], label='AP_50%')
    axs[idx].plot(val_epochs, class_ap[(name, 'AP_25')], label='AP_25%')
    axs[idx].set_ylabel(name)
    axs[idx].legend()
    axs[idx].grid(True)

axs[-1].set_xlabel('Epoch')
plt.suptitle('Per-Class Validation AP Over Epochs')
plt.tight_layout()
plt.show()