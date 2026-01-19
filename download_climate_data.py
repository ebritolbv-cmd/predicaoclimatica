# -*- coding: utf-8 -*-
import cdsapi
import os
import sys

# ==============================================================================
# INSTRUÇÕES DE USO:
# 1. Instale a biblioteca: sudo pip3 install cdsapi
# 2. Configure suas credenciais da CDS API.
#    Crie um arquivo chamado .cdsapirc no seu diretório home (/home/ubuntu/)
#    com o seguinte conteúdo (substitua USER_ID e API_KEY pelos seus dados):
#
#    url: https://cds.climate.copernicus.eu/api/v1
#    key: USER_ID:API_KEY
#
# 3. Execute o script: python3 download_climate_data.py
# ==============================================================================

# Coordenadas de Uberlândia (aproximadas: -18.91, -48.27)
# A área é definida como [Norte, Oeste, Sul, Leste]
UBERLANDIA_AREA = [
    -18.0,  # Norte
    -49.0,  # Oeste
    -19.0,  # Sul
    -48.0,  # Leste
]

# Variáveis climáticas para o modelo
ERA5_VARIABLES = [
    '2m_temperature',
    '2m_dewpoint_temperature', # Usado para calcular umidade relativa
    'surface_pressure',
    'surface_solar_radiation_downwards',
]

# Variáveis para SST do Atlântico Sul (para teleconexão)
SST_AREA = [
    0,      # Norte (Equador)
    -50,    # Oeste
    -30,    # Sul
    -10,    # Leste
]
SST_VARIABLE = ['sea_surface_temperature']

def download_era5_data(client):
    """Baixa dados de reanálise ERA5 para Uberlândia e SST do Atlântico Sul."""
    print("Iniciando download dos dados ERA5 para Uberlândia...")
    
    # 1. Dados Locais (Uberlândia)
    client.retrieve(
        'reanalysis-era5-single-levels',
        {
            'product_type': 'reanalysis',
            'format': 'netcdf',
            'variable': ERA5_VARIABLES,
            'year': '2020',
            'month': ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12'],
            'day': [f'{i:02d}' for i in range(1, 32)],
            'time': ['12:00'], # Exemplo: 12h UTC
            'area': UBERLANDIA_AREA,
        },
        'era5_uberlandia_2020.nc'
    )
    print("Download de era5_uberlandia_2020.nc concluído.")

    # 2. Dados de Teleconexão (SST Atlântico Sul)
    print("Iniciando download dos dados SST do Atlântico Sul...")
    client.retrieve(
        'reanalysis-era5-single-levels',
        {
            'product_type': 'reanalysis',
            'format': 'netcdf',
            'variable': SST_VARIABLE,
            'year': '2020',
            'month': ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12'],
            'day': ['01'], # Exemplo: Apenas o primeiro dia do mês para um índice mensal
            'time': ['12:00'],
            'area': SST_AREA,
        },
        'era5_sst_atlantic_2020.nc'
    )
    print("Download de era5_sst_atlantic_2020.nc concluído.")


def download_cmip6_data(client):
    """Baixa dados de projeção CMIP6 para cenários SSP2-4.5 e SSP5-8.5."""
    print("\nIniciando download dos dados CMIP6 (SSP2-4.5 e SSP5-8.5)...")
    
    # Exemplo de requisição para o cenário SSP2-4.5
    client.retrieve(
        'projections-cmip6',
        {
            'format': 'netcdf',
            'temporal_resolution': 'monthly',
            'experiment': 'ssp2_4_5',
            'model': 'ipsl_cm6a_lr', # Exemplo de modelo
            'variable': 'near_surface_air_temperature',
            'year': ['2040', '2041'],
            'month': ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12'],
            'area': UBERLANDIA_AREA,
        },
        'cmip6_ssp245_uberlandia_2040_2041.nc'
    )
    print("Download de cmip6_ssp245_uberlandia_2040_2041.nc concluído.")

    # Exemplo de requisição para o cenário SSP5-8.5
    client.retrieve(
        'projections-cmip6',
        {
            'format': 'netcdf',
            'temporal_resolution': 'monthly',
            'experiment': 'ssp5_8_5',
            'model': 'ipsl_cm6a_lr', # Exemplo de modelo
            'variable': 'near_surface_air_temperature',
            'year': ['2040', '2041'],
            'month': ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12'],
            'area': UBERLANDIA_AREA,
        },
        'cmip6_ssp585_uberlandia_2040_2041.nc'
    )
    print("Download de cmip6_ssp585_uberlandia_2040_2041.nc concluído.")


if __name__ == '__main__':
    try:
        # Inicializa o cliente CDS API
        c = cdsapi.Client()
        
        # Executa os downloads
        download_era5_data(c)
        download_cmip6_data(c)
        
        print("\nTodos os downloads foram solicitados com sucesso.")
        print("Os arquivos .nc (NetCDF) estão prontos para processamento.")

    except Exception as e:
        print(f"Erro ao executar o script: {e}")
        print("\nVerifique se a biblioteca 'cdsapi' está instalada e se o arquivo '.cdsapirc' está configurado corretamente no seu diretório home.")
        sys.exit(1)
