import os
import time
import pandas as pd
from MAHV_ger_col import MaxHVColumnGeneration  # Importa a classe de Geração de Colunas
from MAHV_ger_col import MaxHVOriginalFormulation  # Importa a classe de Formulação Original
from readershappyinstances import GraphInstance

def get_files_in_folder(folder_path, extension=".txt"):
    files = [f for f in os.listdir(folder_path) if f.endswith(extension)]
    full_paths = [os.path.join(folder_path, f) for f in files]
    return full_paths

def create_log_folder():
    log_folder = '/home/rafael/Downloads/GerCol'
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)
    return log_folder

# Função para salvar resultados no Excel na primeira linha disponível
def save_results_to_excel(results, log_folder, filename="results_gercol.xlsx"):
    excel_file_path = os.path.join(log_folder, filename)

    # Verifica se o arquivo Excel existe
    if os.path.exists(excel_file_path):
        # Carrega o arquivo Excel existente
        df = pd.read_excel(excel_file_path)

        # Concatena os resultados na primeira linha disponível
        new_results = pd.DataFrame(results)
        df = pd.concat([df, new_results], ignore_index=True)
    else:
        # Se o arquivo não existir, cria um DataFrame novo com as colunas
        df = pd.DataFrame(results)
    
    # Salva os resultados no arquivo Excel
    df.to_excel(excel_file_path, index=False)
    print(f"Resultados salvos em: {excel_file_path}")

def process_files_in_folder(folder_path):
    files = get_files_in_folder(folder_path)
    results = []
    log_folder = create_log_folder()  # Cria a pasta de logs

    for file in files:
        print(f"\nProcessando arquivo: {file}")
        instancia = GraphInstance(file)
        adj_list, precolored, vertices, arestas, k_val = instancia.get_data()  # Lê dados do arquivo

        instance_name = os.path.basename(file)
        print(f"\nValor de k: {k_val}")
        log_file_path = os.path.join(log_folder, f'log_{instance_name}_k{k_val}.txt')

        # Resolver com a Geração de Colunas com tolerância de 10 segundos
        maxhv_column_gen = MaxHVColumnGeneration(adj_list, precolored, vertices, k_val)
        start_time = time.time()
        columns, objective_value_column_gen = None, None
        while True:
            elapsed_time = time.time() - start_time
            if elapsed_time > 10:  # Se passou mais de 10 segundos, interrompe
                print(f"Tempo limite de 10 segundos excedido para Geração de Colunas na instância {instance_name}!")
                break
            columns, objective_value_column_gen = maxhv_column_gen.column_generation()
            if columns:  # Se convergiu ou encontrou uma solução válida
                break
        exec_time_column_gen = time.time() - start_time

        # Resolver com a Formulação Original com tolerância de 10 segundos
        maxhv_orig = MaxHVOriginalFormulation(adj_list, precolored, vertices, k_val)
        start_time = time.time()
        solution_values_orig, objective_value_orig = None, None
        while True:
            elapsed_time = time.time() - start_time
            if elapsed_time > 10:  # Se passou mais de 10 segundos, interrompe
                print(f"Tempo limite de 10 segundos excedido para Formulação Original na instância {instance_name}!")
                break
            solution_values_orig, objective_value_orig = maxhv_orig.solve()
            if solution_values_orig:  # Se convergiu ou encontrou uma solução válida
                break
        exec_time_orig = time.time() - start_time

        # Armazenar resultados para a instância
        results.append({
            "instance_name": instance_name,
            "vertices": len(vertices),
            "k": k_val,
            "exec_time_column_gen": exec_time_column_gen,
            "objective_value_column_gen": objective_value_column_gen if objective_value_column_gen else "Não Convergiu",
            "exec_time_orig": exec_time_orig,
            "objective_value_orig": objective_value_orig if objective_value_orig else "Não Convergiu"
        })

        # Salvar os resultados em um arquivo Excel após cada instância
        save_results_to_excel(results, log_folder)

        # Limpar a lista de resultados para a próxima instância
        results.clear()

# Exemplo de uso:
if __name__ == "__main__":
    folder_path = '/home/rafael/Documents/HappySet/MIHS/inputs/happygen/output/testes/testes_cp_benders'  
    process_files_in_folder(folder_path)
