import os
import time
import pandas as pd
from contextlib import contextmanager
from readershappyinstances import GraphInstance
from relax_cut_MHIS import RelaxAndCutMHIS
from MHIS_BC_v2 import modelo_MHIS_BC
import openpyxl
import logging
import sys


# Configurações de log
def setup_logger(name, log_file, level=logging.INFO):
    """Configura o logger para gravar mensagens em arquivos."""
    logger = logging.getLogger(name)
    handler = logging.FileHandler(log_file)
    handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.setLevel(level)
    logger.addHandler(handler)
    return logger


@contextmanager
def redirect_output(log_file):
    """Redireciona stdout e stderr para um arquivo de log."""
    original_stdout, original_stderr = sys.stdout, sys.stderr
    with open(log_file, 'w') as log:
        sys.stdout, sys.stderr = log, log
        try:
            yield
        finally:
            sys.stdout, sys.stderr = original_stdout, original_stderr


def execute_mhis_branch_and_cut(vertices, edges, precolored):
    """Executa Branch-and-Cut para uma instância."""
    try:
        valor_otimo, resultado, tempo_execucao, tempo_prob_sep, num_user_cuts, num_nodes_processed, mip_gap, _ = modelo_MHIS_BC(vertices, edges, precolored)
        return {
            "otimo_bc": valor_otimo,
            "tempo_bc": tempo_execucao,
            "resultado_bc": resultado,
            "tempo_prob_sep_bc": tempo_prob_sep,
            "num_user_cuts": num_user_cuts,
            "num_nodes_processed": num_nodes_processed,
            "gap_bc": mip_gap,
        }
    except Exception as e:
        print(f"Erro em Branch-and-Cut: {e}")
        return None


def execute_relax_and_cut(filename):
    """Executa Relax-and-Cut para uma instância."""
    try:
        modelo = RelaxAndCutMHIS(filename)
        start_time = time.time()
        optimal_value = modelo.solve()
        end_time = time.time()
        tempo_execucao = end_time - start_time
        return {
            "otimo_rc": optimal_value,
            "tempo_rc": tempo_execucao,
        }
    except Exception as e:
        print(f"Erro em Relax-and-Cut: {e}")
        return None


def save_results_to_excel(results, output_file):
    """Salva os resultados consolidados em um arquivo Excel."""
    headers = [
        "Instance", "Vertices", "Edges", "Precolored", "Precolored_Number",
        "Time_B&C", "Optimal_Value_B&C", "GAP_B&C",
        "Time_R&C", "Optimal_Value_R&C"
    ]

    df_new = pd.DataFrame(results, columns=headers)

    if os.path.exists(output_file):
        df_existing = pd.read_excel(output_file)
        df_consolidated = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_consolidated = df_new

    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        df_consolidated.to_excel(writer, index=False)

    print(f"Resultados salvos em {output_file}")


def process_files_in_folder(folder_path, output_file):
    """Processa todas as instâncias da pasta e salva os resultados."""
    files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.txt')]
    results = []
    log_folder = os.path.join(folder_path, "logs")
    os.makedirs(log_folder, exist_ok=True)

    for file in files:
        print(f"\nProcessando instância: {file}")
        instancia = GraphInstance(file)
        adj_list, precolored, vertices, edges, k = instancia.get_data()
        log_file_bc = os.path.join(log_folder, f"log_{os.path.basename(file)}_bc.log")
        log_file_rc = os.path.join(log_folder, f"log_{os.path.basename(file)}_rc.log")

        # Executar Branch-and-Cut
        with redirect_output(log_file_bc):
            bc_results = execute_mhis_branch_and_cut(vertices, edges, precolored)

        # Executar Relax-and-Cut
        with redirect_output(log_file_rc):
            rc_results = execute_relax_and_cut(file)

        # Consolidar resultados
        if bc_results and rc_results:
            results.append([
                file, len(vertices), len(edges),
                precolored, len(precolored),
                bc_results["tempo_bc"], bc_results["otimo_bc"], bc_results["gap_bc"],
                rc_results["tempo_rc"], rc_results["otimo_rc"]
            ])
        else:
            print(f"Erro ao processar {file}. Resultados incompletos.")

    # Salvar resultados no Excel
    save_results_to_excel(results, output_file)


if __name__ == "__main__":
    folder_path = '/home/rafael/Documents/HappySet/MIHS/inputs/happygen/output/testes'
    folder_path = '/home/rafael/Documents/HappySet/MIHS/inputs/happygen/output/testes/20-50-100 (reduce)'
    output_file = '/home/rafael/Downloads/MHIS_Results_Comparison.xlsx'
    process_files_in_folder(folder_path, output_file)
