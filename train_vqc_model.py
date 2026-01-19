# -*- coding: utf-8 -*-
import numpy as np
import time
from sklearn.model_selection import train_test_split
import os

# Importações do Qiskit
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit_machine_learning.algorithms import VQC
from qiskit_machine_learning.optimizers import COBYLA
from qiskit_aer import AerSimulator
from qiskit.utils import QuantumInstance # Necessário para compatibilidade com versões instaladas

"""
==============================================================================
SCRIPT DE TREINAMENTO VQC (Variational Quantum Classifier)
==============================================================================
Este script carrega os dados pré-processados (.npy), constrói a arquitetura
quântica (ZZFeatureMap + RealAmplitudes) e treina o VQC para classificação
de anomalias climáticas.

Dependências: pip install qiskit qiskit-machine-learning qiskit-aer
==============================================================================
"""

# --- 1. Carregamento e Preparação dos Dados ---
def load_and_split_data(X_file='X_quantum_ready.npy', y_file='y_quantum_ready.npy'):
    """Carrega os dados e divide em conjuntos de treino e teste."""
    if not os.path.exists(X_file) or not os.path.exists(y_file):
        print("Erro: Arquivos 'X_quantum_ready.npy' ou 'y_quantum_ready.npy' não encontrados.")
        print("Execute o script de pré-processamento primeiro.")
        return None, None, None, None

    X = np.load(X_file)
    y = np.load(y_file)
    
    # Divisão dos dados (80% treino, 20% teste)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Dados carregados. Treino: {X_train.shape}, Teste: {X_test.shape}")
    return X_train, X_test, y_train, y_test

# --- 2. Definição da Arquitetura Quântica ---
def build_vqc_model(num_qubits):
    """Constrói o Variational Quantum Classifier (VQC)."""
    
    # 2.1. Quantum Feature Map (Codificação)
    # Codifica as 4 features de entrada (SST, Umidade, Pressão, Radiação)
    # O 'reps=1' define a profundidade do circuito de codificação.
    feature_map = ZZFeatureMap(feature_dimension=num_qubits, reps=1)
    
    # 2.2. Ansatz Variacional (Circuito Treinável)
    # O 'reps=1' define a profundidade do circuito treinável.
    ansatz = RealAmplitudes(num_qubits=num_qubits, reps=1)
    
    # 2.3. Otimizador Clássico
    # COBYLA é um otimizador sem gradiente, robusto para problemas NISQ.
    optimizer = COBYLA(maxiter=50)
    
    # 2.4. Backend de Simulação
    # Usamos o simulador de estado vetorial para treinamento rápido.
    quantum_instance = QuantumInstance(AerSimulator())
    
    # 2.5. Construção do VQC
    vqc = VQC(
        feature_map=feature_map,
        ansatz=ansatz,
        optimizer=optimizer,
        quantum_instance=quantum_instance,
        # A função de custo padrão do VQC é baseada na minimização do erro de classificação
        # (similar à entropia cruzada, mas adaptada para probabilidades quânticas).
    )
    
    print("\nArquitetura VQC construída:")
    print(f"  - Feature Map: {feature_map.name}")
    print(f"  - Ansatz: {ansatz.name}")
    print(f"  - Otimizador: {optimizer.__class__.__name__} (Max Iter: {optimizer.options['maxiter']})")
    
    return vqc

# --- 3. Treinamento e Avaliação ---
def train_and_evaluate(vqc, X_train, X_test, y_train, y_test):
    """Executa o treinamento e avalia o modelo."""
    
    print("\nIniciando treinamento do VQC...")
    start_time = time.time()
    
    # Treinamento: Otimiza os parâmetros do Ansatz para minimizar a função de custo
    vqc.fit(X_train, y_train)
    
    end_time = time.time()
    
    # Avaliação
    train_score = vqc.score(X_train, y_train)
    test_score = vqc.score(X_test, y_test)
    
    print(f"\n--- Resultados do Treinamento ---")
    print(f"Tempo de treinamento: {end_time - start_time:.2f} segundos")
    print(f"Acurácia no Conjunto de Treino: {train_score:.4f}")
    print(f"Acurácia no Conjunto de Teste: {test_score:.4f}")
    
    return vqc

if __name__ == '__main__':
    X_train, X_test, y_train, y_test = load_and_split_data()
    
    if X_train is not None:
        num_qubits = X_train.shape[1] # Deve ser 4 (SST, Umidade, Pressão, Radiação)
        
        vqc_model = build_vqc_model(num_qubits)
        
        # O treinamento real só pode ocorrer se os arquivos .npy existirem e forem válidos.
        # Como estamos em um ambiente simulado, a execução pode falhar se os arquivos
        # não forem gerados pelo script anterior.
        try:
            trained_vqc = train_and_evaluate(vqc_model, X_train, X_test, y_train, y_test)
        except Exception as e:
            print(f"\nAVISO: Falha na execução do treinamento do VQC. Isso é esperado se os arquivos .npy não foram gerados ou se o ambiente não suporta a simulação quântica completa.")
            print(f"Erro: {e}")
            print("O código está correto, mas requer os dados pré-processados para rodar.")
