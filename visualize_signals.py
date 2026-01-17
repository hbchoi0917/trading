import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import glob

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.facecolor'] = 'white'

# Find the most recent signals file
signal_files = glob.glob('signals_*.csv')
if not signal_files:
    print("❌ No signal files found. Run the screener first.")
    exit()

latest_file = max(signal_files)
print(f"📊 Analyzing: {latest_file}")

# Read the signals
df = pd.read_csv(latest_file, index_col='Ticker')

if df.empty:
    print("❌ No signals found in the file.")
    exit()

print(f"✓ Found {len(df)} signals to analyze")

# Create comprehensive visualization
fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# ============ 1. SIGNAL STRENGTH RANKING ============
ax1 = fig.add_subplot(gs[0, :2])
top_10 = df.nsmallest(10, 'RSI').sort_values('Signal_Strength', ascending=True)
colors = plt.cm.RdYlGn(top_10['Signal_Strength'] / 100)
ax1.barh(range(len(top_10)), top_10['Signal_Strength'], color=colors)
ax1.set_yticks(range(len(top_10)))
ax1.set_yticklabels(top_10.index)
ax1.set_xlabel('Signal Strength Score', fontsize=11, fontweight='bold')
ax1.set_title('Top 10 Signals by Strength (100 = Highest Quality)', fontsize=12, fontweight='bold')
ax1.set_xlim(0, 100)
for i, (idx, row) in enumerate(top_10.iterrows()):
    ax1.text(row['Signal_Strength'] + 1, i, f"{row['Signal_Strength']:.0f}", 
             va='center', fontsize=9, fontweight='bold')

# ============ 2. SUMMARY STATS ============
ax2 = fig.add_subplot(gs[0, 2])
ax2.axis('off')
stats_text = f"""
SCREENING SUMMARY
{'='*25}

Total Signals: {len(df)}
Scan Date: {df['Scan_Date'].iloc[0]}

QUALITY METRICS
{'='*25}
Avg Signal Strength: {df['Signal_Strength'].mean():.1f}/100
Avg RSI: {df['RSI'].mean():.1f}
Avg ATR: {df['ATR_%'].mean():.1f}%
Avg Vol Surge: {df['Vol_Surge'].mean():.1f}x

PRICE RANGE
{'='*25}
Min: ${df['Price'].min():.2f}
Median: ${df['Price'].median():.2f}
Max: ${df['Price'].max():.2f}
"""
ax2.text(0.05, 0.95, stats_text, transform=ax2.transAxes, 
         fontsize=10, verticalalignment='top', family='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

# ============ 3. RSI DISTRIBUTION ============
ax3 = fig.add_subplot(gs[1, 0])
ax3.hist(df['RSI'], bins=15, color='#FF6B6B', edgecolor='black', alpha=0.7)
ax3.axvline(df['RSI'].mean(), color='blue', linestyle='--', linewidth=2, 
            label=f'Mean: {df["RSI"].mean():.1f}')
ax3.axvline(30, color='red', linestyle=':', linewidth=2, label='Oversold (30)')
ax3.set_xlabel('RSI Value', fontsize=10, fontweight='bold')
ax3.set_ylabel('Frequency', fontsize=10, fontweight='bold')
ax3.set_title('RSI Distribution', fontsize=11, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

# ============ 4. BOLLINGER BAND POSITION ============
ax4 = fig.add_subplot(gs[1, 1])
ax4.hist(df['BB_Position'], bins=15, color='#4ECDC4', edgecolor='black', alpha=0.7)
ax4.axvline(0.5, color='gray', linestyle='--', linewidth=2, label='Middle (0.5)')
ax4.axvline(df['BB_Position'].mean(), color='red', linestyle='--', linewidth=2,
            label=f'Mean: {df["BB_Position"].mean():.2f}')
ax4.set_xlabel('BB Position (0=Lower, 1=Upper)', fontsize=10, fontweight='bold')
ax4.set_ylabel('Frequency', fontsize=10, fontweight='bold')
ax4.set_title('Bollinger Band Position', fontsize=11, fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)

# ============ 5. ATR % DISTRIBUTION ============
ax5 = fig.add_subplot(gs[1, 2])
ax5.hist(df['ATR_%'], bins=15, color='#95E1D3', edgecolor='black', alpha=0.7)
ax5.axvline(df['ATR_%'].mean(), color='red', linestyle='--', linewidth=2,
            label=f'Mean: {df["ATR_%"].mean():.1f}%')
ax5.set_xlabel('ATR as % of Price', fontsize=10, fontweight='bold')
ax5.set_ylabel('Frequency', fontsize=10, fontweight='bold')
ax5.set_title('Average True Range (Volatility)', fontsize=11, fontweight='bold')
ax5.legend()
ax5.grid(True, alpha=0.3)

# ============ 6. VOLUME SURGE ANALYSIS ============
ax6 = fig.add_subplot(gs[2, 0])
ax6.scatter(df['Vol_Surge'], df['Signal_Strength'], alpha=0.6, s=100, c=df['RSI'], 
            cmap='RdYlGn_r', edgecolors='black', linewidth=0.5)
ax6.axvline(1.5, color='red', linestyle='--', linewidth=1, alpha=0.5, label='1.5x threshold')
ax6.set_xlabel('Volume Surge Ratio', fontsize=10, fontweight='bold')
ax6.set_ylabel('Signal Strength', fontsize=10, fontweight='bold')
ax6.set_title('Volume Surge vs Signal Quality', fontsize=11, fontweight='bold')
ax6.legend()
ax6.grid(True, alpha=0.3)
cbar = plt.colorbar(ax6.collections[0], ax=ax6)
cbar.set_label('RSI', fontsize=9)

# ============ 7. PRICE vs SMA 200 ============
ax7 = fig.add_subplot(gs[2, 1])
ax7.scatter(df['SMA_200'], df['Price'], alpha=0.6, s=100, c=df['Signal_Strength'], 
            cmap='RdYlGn', edgecolors='black', linewidth=0.5)
min_val = min(df['SMA_200'].min(), df['Price'].min())
max_val = max(df['SMA_200'].max(), df['Price'].max())
ax7.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, alpha=0.5, 
         label='Price = SMA')
ax7.set_xlabel('200-Day SMA ($)', fontsize=10, fontweight='bold')
ax7.set_ylabel('Current Price ($)', fontsize=10, fontweight='bold')
ax7.set_title('Price vs Long-term Trend', fontsize=11, fontweight='bold')
ax7.legend()
ax7.grid(True, alpha=0.3)
cbar = plt.colorbar(ax7.collections[0], ax=ax7)
cbar.set_label('Signal Strength', fontsize=9)

# ============ 8. SUPPORT DISTANCE ANALYSIS ============
ax8 = fig.add_subplot(gs[2, 2])
ax8.scatter(df['Distance_to_Support_%'], df['Signal_Strength'], alpha=0.6, s=100, 
            c=df['BB_Position'], cmap='coolwarm', edgecolors='black', linewidth=0.5)
ax8.axvline(5, color='red', linestyle='--', linewidth=1, alpha=0.5, label='5% threshold')
ax8.set_xlabel('Distance to Support (%)', fontsize=10, fontweight='bold')
ax8.set_ylabel('Signal Strength', fontsize=10, fontweight='bold')
ax8.set_title('Proximity to Support Level', fontsize=11, fontweight='bold')
ax8.legend()
ax8.grid(True, alpha=0.3)
cbar = plt.colorbar(ax8.collections[0], ax=ax8)
cbar.set_label('BB Position', fontsize=9)

# Main title
fig.suptitle(f'Options Premium Screener Analysis - {datetime.now().strftime("%Y-%m-%d")}', 
             fontsize=16, fontweight='bold', y=0.995)

# Save
output_filename = f'signal_analysis_{datetime.now().strftime("%Y%m%d")}.png'
plt.savefig(output_filename, dpi=300, bbox_inches='tight', facecolor='white')
print(f"\n✅ Visualization saved: {output_filename}")

# Display summary
print("\n" + "="*50)
print("ANALYSIS COMPLETE")
print("="*50)
print(f"Signals Analyzed: {len(df)}")
print(f"Average Signal Strength: {df['Signal_Strength'].mean():.1f}/100")
print(f"Best Signal: {df['Signal_Strength'].max():.0f}/100 ({df['Signal_Strength'].idxmax()})")
print(f"Most Oversold: RSI {df['RSI'].min():.1f} ({df['RSI'].idxmin()})")
print("="*50)
