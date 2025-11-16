import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

tnow = pd.Timestamp.now(tz='UTC')
tnow = tnow.replace(hour=0, minute=0, second=0, microsecond=0)
xlim_start = tnow - pd.Timedelta(days=6)
xlim_end   = tnow + pd.Timedelta(days=1)

def custom_formatter(x, pos):
    ticks = ax.get_xticks()
    if len(ticks) == 0:
        return ""
    
    # Last tick (end of day)
    if abs(x - ticks[-1]) < 1e-6:
        dt = mdates.num2date(x)
        ym = dt.strftime("%Y-%m")
        d = dt.strftime("%d")
        return f"{d}\n{ym}"
    
    # Other ticks (day of month)
    dt = mdates.num2date(x)
    return dt.strftime("%d")

# Data Download

# R ---------------------------------------------------------------
df = pd.read_json(
    'https://services.swpc.noaa.gov/json/goes/primary/xrays-7-day.json'
)
sat = df['satellite'].unique()
if len(sat) == 1:
    sat = sat[0]
    print(f"GOES satellite: {sat}")
else:
    print(f"Multiple satellites found: {sat}")
    sat = sat[0]
df = df.pivot(
    index='time_tag',
    columns='energy',
    values='flux'
)
df.index = pd.to_datetime(
    df.index,
    format='%Y-%m-%dT%H:%M:%SZ'
)
# Plotting
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(1, 1, 1)
ax.plot(
    df.index,
    df['0.1-0.8nm'],
    label='Long (1.0 - 8.0 Å)',
    color='red',
)
ax.legend(fontsize=16, loc='upper left')
ax.fill_between(
    df.index,
    df['0.1-0.8nm'],
    color='red',
    alpha=0.1,
)
ax.set_ylabel('X-ray Flux [W/m$^2$]', fontsize=20)
ax.set_xlabel('Time (UTC)', fontsize=20)
ax.set_title(f'GOES-{sat} X-ray Flux (NOAA/SWPC)', fontsize=20)
ax.set_yscale('log')
ax.set_ylim(1e-9, 1e-2)
ax.set_yticks(np.logspace(-9, -2, num=8))
ax.tick_params(axis='both', which='major', labelsize=20, length=16, width=1)
ax.tick_params(axis='both', which='minor', labelsize=20, length=8, width=1)
ax.tick_params(axis='y', which='both', right=True)
ax.grid(True, which='major', axis='y', linestyle='-', alpha=0.8)

ax.xaxis.set_major_formatter(custom_formatter)
ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
ax.set_xlim(xlim_start, xlim_end)

ax.axhline(1e-5, color='gold', linestyle='--', zorder=2)
ax.text(1.03, 1e-5, 'R1', color='gold', fontsize=20, 
        va='center', transform=ax.get_yaxis_transform())
ax.axhline(5e-5, color='orange', linestyle='--', zorder=2)
ax.text(1.03, 5e-5, 'R2', color='orange', fontsize=20, 
        va='center', transform=ax.get_yaxis_transform())
ax.axhline(1e-4, color='darkorange', linestyle='--', zorder=2)
ax.text(1.03, 1e-4, 'R3', color='darkorange', fontsize=20, 
        va='center', transform=ax.get_yaxis_transform())
ax.axhline(1e-3, color='red', linestyle='--', zorder=2)
ax.text(1.03, 1e-3, 'R4', color='red', fontsize=20, 
        va='center', transform=ax.get_yaxis_transform())
ax.axhline(2e-3, color='darkred', linestyle='--', zorder=2)
ax.text(1.03, 2e-3, 'R5', color='darkred', fontsize=20, 
        va='center', transform=ax.get_yaxis_transform())
ax.text(1.12, 2e-6, 'Radio Blackout Scale', rotation=90, fontsize=20, 
        va='center', ha='center', transform=ax.get_yaxis_transform())

ax.text(0.97, 1e-8, 'A', fontsize=20, va='bottom', ha='center', transform=ax.get_yaxis_transform(), 
        color='black')
ax.text(0.97, 1e-7, 'B', fontsize=20, va='bottom', ha='center', transform=ax.get_yaxis_transform(), 
        color='black')
ax.text(0.97, 1e-6, 'C', fontsize=20, va='bottom', ha='center', transform=ax.get_yaxis_transform(), 
        color='black')
ax.text(0.97, 1e-5, 'M', fontsize=20, va='bottom', ha='center', transform=ax.get_yaxis_transform(), 
        color='black')
ax.text(0.97, 1e-4, 'X', fontsize=20, va='bottom', ha='center', transform=ax.get_yaxis_transform(), 
        color='black')

fig.tight_layout()
plt.savefig('goes_xray_flux.png', dpi=300, bbox_inches='tight')
plt.close(fig)


# S ---------------------------------------------------------------
df = pd.read_json(
    'https://services.swpc.noaa.gov/json/goes/primary/integral-protons-7-day.json'
)
sat = df['satellite'].unique()
if len(sat) == 1:
    sat = sat[0]
else:
    print(f"Multiple satellites found: {sat}")
    sat = sat[0]
df = df.pivot(
    index='time_tag',
    columns='energy',
    values='flux'
)
df.index = pd.to_datetime(
    df.index,
    format='%Y-%m-%dT%H:%M:%SZ'
)
# Plotting
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(1, 1, 1)
ax.plot(
    df.index, 
    df['>=10 MeV'],
    label=r'$\geq 10 \text{ MeV}$',
    color='red',
)
ax.legend(fontsize=16, loc='upper left')
ax.fill_between(
    df.index,
    df['>=10 MeV'],
    color='red',
    alpha=0.1,
)
ax.set_ylabel('Proton Flux [particles/(cm$^2$ s sr)]', fontsize=20)
ax.set_xlabel('Time (UTC)', fontsize=20)
ax.set_title(f'GOES-{sat} Proton Flux (NOAA/SWPC)', fontsize=20)
ax.set_yscale('log')
ax.set_ylim(1e-1, 5e5)
ax.set_yticks(np.logspace(-1, 5, 7))
ax.tick_params(axis='both', which='major', labelsize=20, length=16, width=1)
ax.tick_params(axis='both', which='minor', labelsize=20, length=8, width=1)
ax.tick_params(axis='y', which='both', right=True)
ax.grid(True, which='major', axis='y', linestyle='-', alpha=0.8)

ax.xaxis.set_major_formatter(custom_formatter)
ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
ax.set_xlim(xlim_start, xlim_end)

ax.axhline(1e1, color='gold', linestyle='--', zorder=2)
ax.text(1.03, 1e1, 'S1', color='gold', fontsize=20, 
        va='center', transform=ax.get_yaxis_transform())
ax.axhline(1e2, color='orange', linestyle='--', zorder=2)
ax.text(1.03, 1e2, 'S2', color='orange', fontsize=20, 
        va='center', transform=ax.get_yaxis_transform())
ax.axhline(1e3, color='darkorange', linestyle='--', zorder=2)
ax.text(1.03, 1e3, 'S3', color='darkorange', fontsize=20, 
        va='center', transform=ax.get_yaxis_transform())
ax.axhline(1e4, color='red', linestyle='--', zorder=2)
ax.text(1.03, 1e4, 'S4', color='red', fontsize=20, 
        va='center', transform=ax.get_yaxis_transform())
ax.axhline(1e5, color='darkred', linestyle='--', zorder=2)
ax.text(1.03, 1e5, 'S5', color='darkred', fontsize=20, 
        va='center', transform=ax.get_yaxis_transform())
ax.text(1.12, 2e2, 'Solar Radiation Storm Scale', rotation=90, fontsize=20, 
        va='center', ha='center', transform=ax.get_yaxis_transform())

fig.tight_layout()
plt.savefig('goes_proton_flux.png', dpi=300, bbox_inches='tight')
plt.close(fig)


# G ---------------------------------------------------------------
url = 'http://www-app3.gfz-potsdam.de/kp_index/Kp_ap_nowcast.txt'
res = requests.get(url)
res.raise_for_status()  # Ensure we notice bad responses

text = res.text
lines = text.splitlines()

data = []
for line in lines:
    if line.startswith('#'):
        if '#YYY' in line:
            data.append(line)
        else:
            continue
    else:
        data.append(line)
header = data[0]
rows = data[1:]
header = header.replace('#', '').strip()

df = pd.DataFrame(
    [row.split() for row in rows],
    columns=header.split()
)
df['date'] = df['YYY'] + '-' + df['MM'] + '-' + df['DD'] + 'T' + df['hh.h']
df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%dT%H.%M')
df['Kp'] = pd.to_numeric(df['Kp'], errors='coerce')
df['Kp'] = df['Kp'].replace(-1, None)
df = df.dropna(subset=['Kp'])
# Plotting
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(1, 1, 1)
bar_width = 3/24
colors = []
for v in df['Kp'].values:
    if v < 4:
        colors.append("lime")
    elif v < 5:
        colors.append("greenyellow")
    elif v < 6:
        colors.append("gold")
    elif v < 7:
        colors.append("orange")
    elif v < 8:
        colors.append("darkorange")
    elif v < 9:
        colors.append("red")
    else:
        colors.append("darkred")
    
ax.bar(df['date'], df['Kp'], width=bar_width, align='edge', color=colors, edgecolor='k')
ax.set_ylabel("Kp Index", fontsize=20)
ax.set_xlabel("Time (UTC)", fontsize=20)
ax.set_title("Kp Index (GFZ)", fontsize=20)
ax.set_ylim(0, 9)
ax.set_yticks(range(0, 10))
ax.tick_params(axis='both', which='major', labelsize=20, length=16, width=1)
ax.tick_params(axis='both', which='minor', labelsize=20, length=8, width=1)
ax.tick_params(axis='y', which='both', right=True)
ax.grid(True, which='major', axis='y', linestyle='-', alpha=0.8)

ax.xaxis.set_major_formatter(custom_formatter)
ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
ax.set_xlim(xlim_start, xlim_end)

ax.axhline(5, color='gold', linestyle='--', zorder=2)
ax.text(1.03, 5, 'G1', color='gold', fontsize=20, 
        va='center', transform=ax.get_yaxis_transform())
ax.axhline(6, color='orange', linestyle='--', zorder=2)
ax.text(1.03, 6, 'G2', color='orange', fontsize=20, 
        va='center', transform=ax.get_yaxis_transform())
ax.axhline(7, color='darkorange', linestyle='--', zorder=2)
ax.text(1.03, 7, 'G3', color='darkorange', fontsize=20, 
        va='center', transform=ax.get_yaxis_transform())
ax.axhline(8, color='red', linestyle='--', zorder=2)
ax.text(1.03, 8, 'G4', color='red', fontsize=20, 
        va='center', transform=ax.get_yaxis_transform())
ax.text(1.03, 9, 'G5', color='darkred', fontsize=20, 
        va='center', transform=ax.get_yaxis_transform())
ax.text(1.12, 4, 'Geomagnetic Storm Scale', rotation=90, fontsize=20, 
        va='center', ha='center', transform=ax.get_yaxis_transform())

fig.tight_layout()
plt.savefig('kp_index.png', dpi=300, bbox_inches='tight')
plt.close(fig)