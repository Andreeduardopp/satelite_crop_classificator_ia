import sys
import os
# Ensure the root project directory is in the Python path so absolute imports work when run directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.data_ingestion.request_sentinel_v1 import SentinelHubService
import io
import re
import tarfile
import sqlite3
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
import numpy as np
import tifffile

class BatchFolderPhenologyPipeline:
    def __init__(self, sentinel_service, output_dir="output"):
        self.service = sentinel_service
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Regex to extract parameters from your filename pattern:
        # CROP_ID_plantio_DD-MM-YY_colheita_DD-MM-YY.kml
        self.filename_regex = re.compile(
            r"([A-ZÁÉÍÓÚÂÊÔÇ]+)_(.+)_plantio_(\d{2}-\d{2}-\d{2})_colheita_(\d{2}-\d{2}-\d{2})\.kml", 
            re.IGNORECASE
        )

        # Custom relative offsets (days post-planting) mapped to each crop's biological milestones
        self.CROP_OFFSETS = {
            "FEIJÃO": [0, 10, 20, 32, 44, 62, 77],        # Rapid 65-90 day lifecycle
            "FEIJAO": [0, 10, 20, 32, 44, 62, 77],
            "SOJA":   [0, 20, 35, 55, 75, 95, 115],       # 90-150 day lifecycle
            "AVEIA":  [0, 20, 40, 65, 85, 105, 125],      # Tailored to tillering & cutting stages
            "MILHO":  [0, 25, 55, 65, 85, 105, 125],      # Mapped to V5, VT, R1, R3, R6 milestones
            "TRIGO":  [0, 20, 45, 65, 85, 110, 130],      # Tracks vegetative tillering vs critical Zadoks phases
            "ARROZ":  [0, 25, 55, 75, 95, 115, 135],      # Captures prolonged vegetative phase & reproduction
            "CAFÉ":   [0, 30, 60, 120, 180, 210, 240],    # Extended 6-8 month fruit development window
            "CAFE":   [0, 30, 60, 120, 180, 210, 240]
        }
        
        # Fallback tracking sequence if a crop class isn't explicitly configured above
        self.DEFAULT_OFFSETS = [0, 20, 40, 60, 80, 100, 120, 140]

    def _parse_metadata_from_filename(self, filename):
        """
        Extracts crop type, planting date, and harvesting date from filename layout.
        Converts dates from DD-MM-YY strings to standard datetime tracking entities.
        """
        match = self.filename_regex.match(filename)
        if not match:
            return None
        
        crop, field_id, plantio_str, colheita_str = match.groups()
        try:
            date_plantio = datetime.strptime(plantio_str, "%d-%m-%y")
            date_colheita = datetime.strptime(colheita_str, "%d-%m-%y")
            
            return {
                'crop_class': crop.upper().strip(),
                'field_id': f"{crop.upper().strip()}_{field_id}",
                'planting_date_str': date_plantio.strftime("%Y-%m-%d"),
                'base_planting_datetime': date_plantio,
                'harvest_date_str': date_colheita.strftime("%Y-%m-%d")
            }
        except ValueError:
            return None

    def _parse_kml_polygon(self, kml_path):
        """Extracts spatial coordinate system from KML boundary file."""
        namespaces = {'kml': 'http://www.opengis.net/kml/2.2'}
        try:
            tree = ET.parse(kml_path)
            root = tree.getroot()
            coord_node = root.find('.//kml:coordinates', namespaces)
            if coord_node is not None:
                coord_str = coord_node.text.strip()
                coords = [[float(p.split(',')[0]), float(p.split(',')[1])] for p in coord_str.split() if len(p.split(',')) >= 2]
                
                # Close the polygon ring if it's open (required by GeoJSON)
                if coords and coords[0] != coords[-1]:
                    coords.append(coords[0])
                    
                if coords:
                    return [coords]
            return None
        except Exception:
            return None

    def process_root_culture_directory(self, root_dir_path, width=100, height=100, max_kml_per_culture=None):
        """
        Iterates over the crop folders, calculates custom date sequences per file, 
        and builds a uniform tabular database output.
        """
        print("🔑 Authenticating API connection...")
        token = self.service.autenticar()
        compiled_dataset = []
        culture_counts = {}
        
        print(f"📁 Scanning root directory: {root_dir_path}")
        for root, dirs, files in os.walk(root_dir_path):
            for file in files:
                if file.lower().endswith('.kml'):
                    file_path = os.path.join(root, file)
                    
                    metadata = self._parse_metadata_from_filename(file)
                    if not metadata:
                        continue
                        
                    crop_key = metadata['crop_class']
                    
                    # Apply max KML limit per culture
                    if max_kml_per_culture is not None:
                        if culture_counts.get(crop_key, 0) >= max_kml_per_culture:
                            continue
                            
                    coordinates = self._parse_kml_polygon(file_path)
                    if not coordinates:
                        continue
                    
                    # Look up the custom biological offset timeline configured for this plant type
                    target_offsets = self.CROP_OFFSETS.get(crop_key, self.DEFAULT_OFFSETS)
                    
                    print(f"🌱 Extracting {crop_key} | Field: {metadata['field_id']} | Using {len(target_offsets)} dynamic offsets.")
                    
                    row_features = {
                        'original_kml_path': file_path,
                        'field_id': metadata['field_id'],
                        'label_crop_class': crop_key,
                        'planting_date': metadata['planting_date_str'],
                        'harvest_date': metadata['harvest_date_str']
                    }
                    
                    base_date = metadata['base_planting_datetime']
                    
                    # Query data loop for each custom developmental offset stage
                    for offset in target_offsets:
                        target_date_start = base_date + timedelta(days=offset)
                        target_date_end = target_date_start + timedelta(days=15)
                        
                        target_date_start_str = target_date_start.strftime("%Y-%m-%d")
                        target_date_end_str = target_date_end.strftime("%Y-%m-%d")
                        
                        try:
                            tar_bytes = self.service.consultar_imagens_tiff(
                                access_token=token, coordenadas=coordinates,
                                data_inicio=target_date_start_str, data_fim=target_date_end_str, 
                                width=width, height=height
                            )
                            
                            with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r") as tar:
                                for member in tar.getmembers():
                                    if member.isfile() and member.name.endswith('.tif'):
                                        idx_name = member.name.split('_')[0] # ndvi, evi, ndwi, ndbi
                                        file_data = tar.extractfile(member).read()
                                        
                                        img_data = tifffile.imread(io.BytesIO(file_data))
                                        if img_data.ndim == 3:
                                            band_arr = img_data[0] if img_data.shape[0] < img_data.shape[-1] else img_data[:, :, 0]
                                        else:
                                            band_arr = img_data
                                            
                                        val = np.nanmedian(band_arr) if not np.all(np.isnan(band_arr)) else np.nan
                                        row_features[f"{idx_name}_day_{offset}"] = val
                                        
                        except Exception as e:
                            print(f"⚠️ Error extracting data for offset {offset}: {e}")
                            # Log safe padding values if an image acquisition fails
                            for idx in ['ndvi', 'evi', 'ndwi', 'ndbi']:
                                row_features[f"{idx}_day_{offset}"] = np.nan
                            
                    compiled_dataset.append(row_features)
                    # Increment counter for this culture after successful processing
                    culture_counts[crop_key] = culture_counts.get(crop_key, 0) + 1

        # Generate final database file output
        output_db_path = os.path.join(self.output_dir, "optimized_phenology_dataset.db")
        
        if compiled_dataset:
            with sqlite3.connect(output_db_path) as conn:
                cursor = conn.cursor()
                
                # Get all unique columns across all rows just in case some failed/skipped
                all_cols_set = set()
                for row in compiled_dataset:
                    all_cols_set.update(row.keys())
                    
                fixed_cols = ['original_kml_path', 'field_id', 'label_crop_class', 'planting_date', 'harvest_date']
                dynamic_cols = sorted([c for c in all_cols_set if c not in fixed_cols])
                all_cols = fixed_cols + dynamic_cols
                
                # Create table
                cols_def = []
                for col in all_cols:
                    if col in fixed_cols:
                        cols_def.append(f'"{col}" TEXT')
                    else:
                        cols_def.append(f'"{col}" REAL')
                        
                # Drop table if exists to ensure schema matches (for testing purposes)
                # In production you might want to alter tables, but for a pipeline output it's easier to create fresh or assume schema consistency.
                cursor.execute(f"CREATE TABLE IF NOT EXISTS phenology_data ({', '.join(cols_def)})")
                
                placeholders = ", ".join(["?"] * len(all_cols))
                cols_str = ", ".join([f'"{c}"' for c in all_cols])
                insert_query = f"INSERT INTO phenology_data ({cols_str}) VALUES ({placeholders})"
                
                for row in compiled_dataset:
                    row_values = [row.get(col, None) for col in all_cols]
                    
                    # Convert numpy floats to native python floats, handle NaNs
                    def sanitize_val(v):
                        if isinstance(v, (float, np.floating)):
                            if np.isnan(v): return None
                            return float(v)
                        return v
                        
                    row_values = [sanitize_val(v) for v in row_values]
                    cursor.execute(insert_query, row_values)
                
                conn.commit()
        
        print(f"\n📊 Extraction complete! Optimized multi-temporal dataset stored at: {output_db_path}")
        return compiled_dataset

# --- How to Run the Script ---
if __name__ == "__main__":  
    sentinel_service = SentinelHubService()
    pipeline = BatchFolderPhenologyPipeline(sentinel_service, output_dir=os.path.join("src", "data", "db"))
    # Running a small test with max_kml_per_culture=2
    pipeline.process_root_culture_directory(os.path.join("src", "data", "dataset_split", "train"), max_kml_per_culture=10)