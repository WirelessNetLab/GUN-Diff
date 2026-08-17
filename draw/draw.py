import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'mathtext.fontset': 'stix',
    'font.size': 14,
    'axes.labelsize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 12,
    'axes.titlesize': 16,
})

data = pd.read_csv(r'draw\data.csv')
Total_loss_train = data['train_loss']
Total_loss_valid = data['valid_loss']
train_cap        = data['cap_train'  ]
valid_cap        = data['cap_valid'  ]

x_loss = np.arange(len(Total_loss_train))
x_cap   = np.arange(len(train_cap))

total_points = len(x_loss)
m_step = max(total_points // 5, 1)
m_offset = m_step // 2

fig, ax1 = plt.subplots()
step = 1


ax1.plot(x_loss[::step], Total_loss_train[::step], 
         linestyle='-', color='#ff7f0e', linewidth=2, 
         marker='o', markersize=8, markevery=m_step,fillstyle='none',
         label='Train Loss')

ax1.plot(x_loss[::step], Total_loss_valid[::step], 
         linestyle='--', color='#1f77b4', linewidth=2, 
         marker='s', markersize=8, markevery=(m_offset, m_step),fillstyle='none',
         label='Valid Loss')

ax1.set_xlabel('Epoch', fontsize=18)
ax1.set_ylabel('Loss', fontsize=18)
ax1.grid(True, linestyle='--', alpha=0.4)

mean_loss = np.mean((Total_loss_train[-100:]+Total_loss_valid[-100:])/2)
ax1.axhline(mean_loss, linestyle='-.', color='gray', linewidth=0.9)
ax1.text(-33, mean_loss, f"{mean_loss:.2f}", ha='right', va='bottom')

ax2 = ax1.twinx()

ax2.plot(x_cap[::step], train_cap[::step],  
         linestyle='-',  linewidth=1.8, color="#AD0000", 
         marker='^', markersize=8, markevery=m_step,fillstyle='none',
         label=r'Train Capacity')

ax2.plot(x_cap[::step], valid_cap[::step],  
         linestyle='--', linewidth=1.8, color="#0515FF", 
         marker='v', markersize=8, markevery=(m_offset, m_step),fillstyle='none',
         label=r'Valid Capacity')

mean_loss = np.mean((train_cap[-50:]+valid_cap[-50:])/2)
ax2.axhline(mean_loss, linestyle='-.', color='gray', linewidth=0.9)
ax2.text(530, mean_loss-0.25, f"{mean_loss:.2f}", ha='left', va='bottom')

ax2.set_ylabel('Network Capacity (bps/Hz)')

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()


ax2.legend(lines2, labels2, loc="center right", bbox_to_anchor=(1., 0.8 ) )
ax1.legend(lines1, labels1, loc="center right", bbox_to_anchor=(1., 0.15) )

plt.tight_layout()
plt.show()
