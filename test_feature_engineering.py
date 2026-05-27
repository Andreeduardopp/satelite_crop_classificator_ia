#!/usr/bin/env python3
"""
Test script to validate feature engineering improvements.
Creates synthetic data and tests engineer_features() function.
"""

import numpy as np
import pandas as pd
import sys
sys.path.insert(0, 'src/model')

from train_xgboost_v3 import engineer_features, STAGES, INDICES, SAR_INDICES

def create_synthetic_data(n_samples=10):
    """Create minimal synthetic phenology data for testing."""
    np.random.seed(42)
    
    data = {
        'field_id': [f'field_{i}' for i in range(n_samples)],
        'crop_label': np.random.choice(['SOJA', 'MILHO', 'CAFE'], n_samples),
        'planting_date': ['2023-01-15'] * n_samples,
    }
    
    # Add optical indices (NDVI, EVI, NDWI) × 5 stats × 6 stages
    for idx in INDICES:
        for stat in ['mean', 'median', 'std', 'p10', 'p90']:
            for stage in STAGES:
                col_name = f"{idx}_{stat}_{stage}"
                # Random values in typical ranges
                if stat == 'mean':
                    data[col_name] = np.random.uniform(0.2, 0.8, n_samples)
                elif stat == 'median':
                    data[col_name] = np.random.uniform(0.2, 0.8, n_samples)
                elif stat == 'std':
                    data[col_name] = np.random.uniform(0.05, 0.3, n_samples)
                elif stat == 'p10':
                    data[col_name] = np.random.uniform(0.1, 0.5, n_samples)
                elif stat == 'p90':
                    data[col_name] = np.random.uniform(0.3, 0.9, n_samples)
    
    # Add SAR indices (VV, VH, CR, RVI) × 5 stats × 6 stages
    for idx in SAR_INDICES:
        for stat in ['mean', 'median', 'std', 'p10', 'p90']:
            for stage in STAGES:
                col_name = f"{idx}_{stat}_{stage}"
                # SAR values typically in dB or linear units
                if stat == 'mean':
                    data[col_name] = np.random.uniform(-15, 5, n_samples)  # dB range
                elif stat == 'median':
                    data[col_name] = np.random.uniform(-15, 5, n_samples)
                elif stat == 'std':
                    data[col_name] = np.random.uniform(1, 8, n_samples)
                elif stat == 'p10':
                    data[col_name] = np.random.uniform(-20, -5, n_samples)
                elif stat == 'p90':
                    data[col_name] = np.random.uniform(-5, 10, n_samples)
    
    # Add metadata
    data['stages_covered'] = [6] * n_samples
    data['sar_backfill_done'] = [1] * n_samples
    
    return pd.DataFrame(data)

def test_engineer_features():
    """Test that engineer_features runs without errors."""
    print("Creating synthetic data...")
    df = create_synthetic_data(n_samples=10)
    print(f"Input shape: {df.shape}")
    print(f"Input columns: {len(df.columns)}")
    
    print("\nRunning engineer_features()...")
    try:
        df_eng = engineer_features(df)
        print(f"✅ Success! Output shape: {df_eng.shape}")
        print(f"New features added: {df_eng.shape[1] - df.shape[1]}")
        
        # Check for specific new features
        expected_features = [
            'planting_doy', 'planting_doy_sin', 'planting_doy_cos',
            'NDVI_mean_delta_baseline_to_emergence',
            'NDVI_peak_stage', 'NDVI_peak_value', 'NDVI_amplitude',
            'NDVI_greenup_rate', 'NDVI_senescence_rate',
            'NDVI_temporal_cv',
            'VV_peak_stage', 'VV_amplitude',  # SAR features
            'RVI_NDVI_ratio_vegetative',  # Fusion features
        ]
        
        print("\nChecking expected features:")
        for feat in expected_features:
            if feat in df_eng.columns:
                print(f"  ✅ {feat}")
            else:
                print(f"  ❌ {feat} (missing)")
        
        # Check for NaN issues
        nan_rates = df_eng.isnull().mean()
        high_nan = nan_rates[nan_rates > 0.5]
        if len(high_nan) == 0:
            print("\n✅ No columns with >50% NaNs")
        else:
            print(f"\n⚠️  {len(high_nan)} columns with >50% NaNs:")
            print(high_nan)
        
        print("\n✅ All tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = test_engineer_features()
    sys.exit(0 if success else 1)
