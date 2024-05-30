import numpy as np
import joblib

# Carregar o modelo treinado
modelo = joblib.load('modelo_multinomial.joblib')

# ID do docente para o qual deseja fazer previsões
id_docente = 1  # Substitua pelo ID do docente desejado

# Criar um array numpy com o ID do docente (reshape para uma única amostra)
novo_docente_ids = np.array([[id_docente]])

# Fazer previsões usando o modelo carregado (obter probabilidades)
probabilidades = modelo.predict_proba(novo_docente_ids)

# Obter todas as classes (ou disciplinas) disponíveis no modelo
classes = modelo.classes_

# Número máximo de disciplinas a serem previstas por docente
num_disciplinas_previstas = 5  # Altere conforme necessário

# Selecionar as disciplinas com as maiores probabilidades
top_indices = np.argsort(probabilidades, axis=1)[:, -num_disciplinas_previstas:]  # Índices das disciplinas mais prováveis

# Ordenar os índices com base nas probabilidades (em ordem decrescente)
sorted_indices = np.flip(top_indices, axis=1)

# Obter as disciplinas previstas ordenadas com base nas probabilidades e suas probabilidades correspondentes
disciplinas_previstas = []
for row in sorted_indices:
    disciplinas = []
    for idx in row:
        disciplina = classes[idx]
        probabilidade = probabilidades[0][idx]
        disciplinas.append((disciplina, probabilidade))
    disciplinas_previstas.append(disciplinas)

# Imprimir as disciplinas previstas ordenadas por probabilidade
print(f"Disciplinas previstas para o docente com ID {id_docente} (ordenadas por probabilidade):")
for i, previsoes in enumerate(disciplinas_previstas):
    print(f"Top {num_disciplinas_previstas} para previsão {i + 1}:")
    for disciplina, probabilidade in previsoes:
        print(f"- Disciplina: {disciplina}, Probabilidade: {probabilidade:.4f}")
