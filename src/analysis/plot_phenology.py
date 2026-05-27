import os
import sqlite3
import re
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def generate_plots(db_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(db_path):
        print(f"Error: Database not found at {db_path}")
        return
        
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        cursor.execute("SELECT * FROM phenology_data")
    except sqlite3.OperationalError:
        print(f"Error: Table 'phenology_data' not found in {db_path}")
        return
        
    columns = [desc[0] for desc in cursor.description]
    rows = cursor.fetchall()
    
    if not rows:
        print("Database is empty. No plots to generate.")
        return
    
    # Organize data: crop -> index -> field_id -> [(day, value)]
    crop_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    
    # Extract day offsets from column names like 'ndvi_day_45'
    day_cols = []
    pattern = re.compile(r"([a-z]+)_day_(\d+)")
    for idx, col in enumerate(columns):
        match = pattern.match(col)
        if match:
            day_cols.append((idx, match.group(1), int(match.group(2))))
            
    field_idx = columns.index("field_id")
    crop_idx = columns.index("label_crop_class")
    
    for row in rows:
        crop = row[crop_idx]
        field = row[field_idx]
        
        for idx, index_name, day in day_cols:
            val = row[idx]
            if val is not None:
                try:
                    crop_data[crop][index_name][field].append((day, float(val)))
                except (ValueError, TypeError):
                    pass
                
    # Calculate median per day for each crop
    # crop_avg_data[index_name][crop] = {day: median_val}
    crop_avg_data = defaultdict(lambda: defaultdict(dict))
    
    for crop, indices in crop_data.items():
        for idx_name, fields in indices.items():
            # Gather all values for a specific day across all fields
            day_values = defaultdict(list)
            for field, data_points in fields.items():
                for day, val in data_points:
                    day_values[day].append(val)
            
            # Compute median for each day
            for day, vals in day_values.items():
                crop_avg_data[idx_name][crop][day] = np.median(vals)
                
    # Set seaborn style for better aesthetics
    sns.set_theme(style="whitegrid")
    
    # Create a single figure comparing all crops
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Comparative Phenological Profiles by Crop (Median Values)', fontsize=20, fontweight='bold', y=0.96)
    
    unique_crops = sorted(crop_data.keys())
    colors = sns.color_palette("tab10", len(unique_crops))
    crop_colors = {crop: color for crop, color in zip(unique_crops, colors)}
    
    for ax, idx_name in zip(axes.flatten(), ['ndvi', 'evi', 'ndwi', 'ndbi']):
        ax.set_title(f"{idx_name.upper()} Profile", fontsize=15, fontweight='semibold')
        ax.set_xlabel("Days After Planting (Offset)", fontsize=13)
        ax.set_ylabel("Median Index Value", fontsize=13)
        
        if idx_name in crop_avg_data:
            for crop, day_dict in crop_avg_data[idx_name].items():
                # Sort data points chronologically by day
                days = sorted(day_dict.keys())
                vals = [day_dict[d] for d in days]
                
                # Plot the average line for this crop
                ax.plot(days, vals, marker='o', linewidth=3, markersize=8, 
                        label=crop, color=crop_colors[crop])
            
            ax.legend(loc='best', fontsize=11, title="Crops")
            
    plt.tight_layout()
    plt.subplots_adjust(top=0.90)
    plot_path = os.path.join(output_dir, "all_crops_comparison.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Generated comparative analysis plot at: {plot_path}")

if __name__ == "__main__":
    db_file = os.path.join("src", "data", "db", "optimized_phenology_dataset.db")
    out_dir = os.path.join("src", "analysis", "plots")
    
    print(f"Analyzing database at: {db_file}")
    generate_plots(db_file, out_dir)
    print("All plotting operations completed.")
