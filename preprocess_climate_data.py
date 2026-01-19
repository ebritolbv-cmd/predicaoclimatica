# -*- coding: utf-8 -*-
import xarray as xr
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os

"""
==============================================================================
SCRIPT DE PRÉ-PROCESSAMENTO: ERA5/CMIP6 PARA QUANTUM MACHINE LEARNING
==============================================================================
Este script processa os arquivos NetCDF (.nc) baixados via CDS API, realiza
o feature engineering necessário e prepara os dados para o ZZFeatureMap.

Dependências: pip install xarray netCDF4 pandas numpy scikit-learn
==============================================================================
"""

def calculate_relative_humidity(t2m, d2m):
    """Calcula a umidade relativa aproximada usando a fórmula de Magnus-Tetens."""
    # t2m e d2m em Kelvin
    T = t2m - 273.15
    TD = d2m - 273.15
    rh = 100 * (np.exp((17.625 * TD) / (243.04 + TD)) / np.exp((17.625 * T) / (243.04 + T)))
    return rh

def preprocess_data(era5_file, sst_file):
    """Lê arquivos NetCDF e prepara o DataFrame final."""
    print(f"Processando arquivos: {era5_file} e {sst_file}")
    
    if not os.path.exists(era5_file) or not os.path.exists(sst_file):
        print("Erro: Arquivos .nc não encontrados. Execute o script de download primeiro.")
        return None

    # 1. Carregar dados de Uberlândia
    ds_uber = xr.open_dataset(era5_file)
    # Extrair médias espaciais se houver mais de um ponto na grade
    df_uber = ds_uber.mean(dim=['latitude', 'longitude']).to_dataframe()
    
    # 2. Calcular Umidade Relativa
    if 't2m' in df_uber.columns and 'd2m' in df_uber.columns:
        df_uber['relative_humidity'] = calculate_relative_humidity(df_uber['t2m'], df_uber['d2m'])
    
    # 3. Carregar dados de SST (Teleconexão)
    ds_sst = xr.open_dataset(sst_file)
    df_sst = ds_sst.mean(dim=['latitude', 'longitude']).to_dataframe()
    
    # 4. Integrar DataFrames (Join pelo tempo)
    # Nota: Pode ser necessário ajustar a frequência (ex: resample para mensal)
    df_final = df_uber.join(df_sst[['sst']], how='inner', rsuffix='_ocean')
    
    # 5. Seleção de Features para o Modelo Quântico (4 Features para 4 Qubits)
    features = ['sst', 'relative_humidity', 'sp', 'ssrd'] # SST, Umidade, Pressão, Radiação
    df_model = df_final[features].dropna()
    
    # 6. Definição do Target (Anomalia Térmica)
    # Exemplo: 1 se a temperatura estiver 2 desvios padrão acima da média
    threshold = df_final['t2m'].mean() + 2 * df_final['t2m'].std()
    df_model['target'] = (df_final['t2m'] > threshold).astype(int)
    
    print(f"Dataset processado com {len(df_model)} amostras.")
    return df_model

def prepare_for_quantum(df):
    """Aplica normalização e prepara arrays para o VQC."""
    features = df.drop(columns=['target'])
    target = df['target']
    
    # Normalização para [0, pi] - Essencial para Quantum Feature Maps
    scaler = MinMaxScaler(feature_range=(0, np.pi))
    X_scaled = scaler.fit_transform(features)
    
    # One-hot encoding para o target (VQC espera 2 saídas para classificação binária)
    y_one_hot = np.zeros((len(target), 2))
    for i, val in enumerate(target):
        y_one_hot[i, val] = 1
        
    return X_scaled, y_one_hot

if __name__ == "__main__":
    # Nomes dos arquivos gerados pelo script de download
    ERA5_FILE = 'era5_uberlandia_2020.nc'
    SST_FILE = 'era5_sst_atlantic_2020.nc'
    
    # Executar processamento
    df_processed = preprocess_data(ERA5_FILE, SST_FILE)
    
    if df_processed is not None:
        X, y = prepare_for_quantum(df_processed)
        
        # Salvar para uso posterior
        np.save('X_quantum_ready.npy', X)
        np.save('y_quantum_ready.npy', y)
        
        print("\nSucesso!")
        print(f"Shape de X (Features): {X.shape}")
        print(f"Shape de y (Target): {y.shape}")
        print("Arquivos 'X_quantum_ready.npy' e 'y_quantum_ready.npy' gerados.")
        
        # Exemplo de visualização das primeiras linhas
        print("\nPrimeiras 5 amostras normalizadas (0 a pi):")
        print(X[:5])
