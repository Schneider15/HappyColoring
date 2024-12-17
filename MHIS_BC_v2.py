#!/usr/bin/python
# from MHIS_INSTANCE_READER import GraphInstance
from readershappyinstances import GraphInstance
import pandas as pd
import re
import time
import sys
import cplex
from cplex.exceptions import CplexError
from cplex.callbacks import UserCutCallback
import networkx as nx
import matplotlib.pyplot as plt
from networkx.algorithms.shortest_paths.weighted import single_source_dijkstra
import openpyxl
import os
import logging

# Diretório para salvar arquivos
directory = "/home/rafael/Documents/HappySet/MIHS"

def save_to_excel(output_file, data, headers=None):
    """Salva dados em um arquivo Excel, criando o arquivo e diretórios se necessário."""
    directory = os.path.dirname(output_file)
    if not os.path.exists(directory):
        os.makedirs(directory)
    if not os.path.exists(output_file):
        workbook = openpyxl.Workbook()
        sheet = workbook.active
        if headers:
            sheet.append(headers)
        workbook.save(output_file)
    
    workbook = openpyxl.load_workbook(output_file)
    sheet = workbook.active
    
    for row in data:
        sheet.append(row)
    
    workbook.save(output_file)
    print(f'Dados adicionados ao arquivo Excel {output_file}')

def setup_logger(name, log_file, level=logging.INFO):
    """Configura o logger para o arquivo especificado."""
    logger = logging.getLogger(name)
    handler = logging.FileHandler(log_file)
    handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.setLevel(level)
    logger.addHandler(handler)
    return logger

class MHISCallback(UserCutCallback):
    def __init__(self, vertices, arestas, precoloridos):
        self.n = len(vertices)
        self.arestas = arestas
        self.precoloridos = precoloridos
        self.lp = -1

    def _gerar_caminhos_violados(self, h_values):
        caminhos = []
        epsilon =0.01
        custos = []
        precolored = list(self.precoloridos.keys())
        G = nx.Graph()

        for (i, j) in self.arestas:
            c_ij = (2+epsilon - h_values[i] - h_values[j]) / 2
            G.add_edge(i, j, weight=c_ij)

        for i in range(len(precolored) - 1):
            for j in range(i + 1, len(precolored)):
                if self.precoloridos[precolored[i]] != self.precoloridos[precolored[j]]:
                    try:
                        caminho = nx.shortest_path(G, source=precolored[i], target=precolored[j], weight='weight')
                        if caminho:
                            caminho_custo = sum(G[u][v]['weight'] for u, v in zip(caminho[:-1], caminho[1:]))
                            if caminho_custo < 1 - 0.01:
                                caminhos.append(caminho)
                                custos.append(caminho_custo)
                    except nx.NetworkXNoPath:
                        continue

        return caminhos, custos

    def invoke(self, context):
        try:
            if context.in_relaxation():
                h_values = context.get_relaxation_point()
                f_objective = context.get_relaxation_objective()
            elif context.in_candidate():
                h_values = context.get_candidate_point()
                f_objective = context.get_candidate_objective()
            else:
                print("Em outro contexto!")
                return

            no_id = context.get_long_info(9)
            #print(h_values)

            caminhos_violados, custos_violados = self._gerar_caminhos_violados(h_values)
            # Incluindo condicional para retornar quando não há caminho violado (não há corte a fazer)
            
            print("no_id: ", no_id, " fobj: ", f_objective, " frac: ", context.in_relaxation(), " int: ", context.in_candidate(), "ncuts: ",len(caminhos_violados))
            if len(caminhos_violados) == 0:
                if no_id == 0:
                    self.lp = f_objective
                return
            
            cuts = []
            rhs = []
            for caminho in caminhos_violados:
                #cut_vars = [f"h_{i}" for i in caminho]
                cut_vals = [1.0] * len(caminho)
                cuts.append(cplex.SparsePair(ind=caminho, val=cut_vals))
                rhs.append(len(caminho) - 1.0)
                #print(caminho, rhs[-1])

            if context.in_relaxation():
                context.add_user_cuts(
                    cuts=cuts,
                    senses=['L'] * len(cuts),
                    rhs=rhs,
                    cutmanagement=[cplex.callbacks.UserCutCallback.use_cut.purge]*len(cuts),
                    local=[False]*len(cuts)
                )
            elif context.in_candidate():
                context.reject_candidate(
                    constraints=cuts,
                    senses=['L'] * len(cuts),
                    rhs=rhs
                )
                        
        except CplexError as e:
            print("Erro no callback:", e)

def modelo_MHIS_BC(vertices, arestas, precoloridos):
    """Configura e resolve o modelo CPLEX com o callback personalizado para cortes."""
    start_time = time.time()

    cpx = cplex.Cplex()
    cpx.set_problem_type(cplex.Cplex.problem_type.LP)
    
    variaveis = [f"h_{i}" for i in range(len(vertices))]
    objetivo = [1] * len(vertices)
    lb = [1 if i in precoloridos else 0 for i in range(len(vertices))]
    
    cpx.variables.add(
        names=variaveis,
        obj=objetivo,
        lb=lb,
        ub=[1] * len(vertices),
        types=[cpx.variables.type.binary] * len(vertices)
    )

    cpx.objective.set_sense(cpx.objective.sense.maximize)

    cpx.parameters.mip.strategy.heuristicfreq.set(-1)
    cpx.parameters.mip.cuts.mircut.set(-1)
    cpx.parameters.mip.cuts.implied.set(-1)
    cpx.parameters.mip.cuts.gomory.set(-1)
    cpx.parameters.mip.cuts.flowcovers.set(-1)
    cpx.parameters.mip.cuts.pathcut.set(-1)
    cpx.parameters.mip.cuts.liftproj.set(-1)
    cpx.parameters.mip.cuts.zerohalfcut.set(-1)
    cpx.parameters.mip.cuts.cliques.set(-1)
    cpx.parameters.mip.cuts.covers.set(-1)
    cpx.parameters.threads.set(1)
    cpx.parameters.clocktype.set(1)
    cpx.parameters.timelimit.set(600)
    cpx.parameters.preprocessing.presolve.set(0)
    cpx.parameters.preprocessing.boundstrength.set(0)
    cpx.parameters.preprocessing.coeffreduce.set(0)
    cpx.parameters.preprocessing.relax.set(0)
    cpx.parameters.preprocessing.aggregator.set(0)
    cpx.parameters.preprocessing.reduce.set(0)
    cpx.parameters.preprocessing.reformulations.set(0)

    start_time_prob_sep = time.time()
    mhis_callback = MHISCallback(vertices, arestas, precoloridos)
    contextmask = cplex.callbacks.Context.id.relaxation | cplex.callbacks.Context.id.candidate
    cpx.set_callback(mhis_callback, contextmask)
    cpx.write("teste.lp")
    cpx.solve()

    end_time_prob_sep = time.time()
    tempo_execucao_prob_sep = end_time_prob_sep - start_time_prob_sep

    end_time = time.time()
    tempo_execucao = end_time - start_time

    # Informações sobre a solução
    status_code = cpx.solution.get_status()
    status_description = cpx.solution.status[status_code]
    print(f'Status da solução:                   {status_description}')
    print('Nós processados:                     %d' % cpx.solution.progress.get_num_nodes_processed())
    # print('Status da solução:                   %d' % cpx.solution.get_status())
    # print('Nós processados:                     %d' % cpx.solution.progress.get_num_nodes_processed())
    # print('Valor ótimo:                         %f' % cpx.solution.get_objective_value())
    print(f"Tempo de execução do Prob Sep: {tempo_execucao_prob_sep} segundos")
    print('Valor ótimo:                         %f' % cpx.solution.get_objective_value())
    # Imprimindo o número de cortes ativos
    num_user_cuts = cpx.solution.MIP.get_num_cuts(cpx.solution.MIP.cut_type.user)
        
    # Captura do GAP
    mip_gap = cpx.solution.MIP.get_mip_relative_gap()
    print(f"GAP relativo MIP:                     {mip_gap * 100:.2f}%")
    
    print('Active user cuts:                    %d' % num_user_cuts)

    valores = cpx.solution.get_values()
    print('Solução ótima:', ' '.join(f"h_{i}={valores[i]}" for i in range(len(vertices))))
    # for i in range(len(vertices)):
    #     print(f"h_{i} = {valores[i]}")
    print(f"Tempo total de execução: {tempo_execucao} segundos")

    print("Valor LP:", mhis_callback.lp)

    # Grava resultados no Excel
    # result_data = [
    #     [self.filename, 'MHIS_BC', self.n, len(self.arestas), total_time, optimal_value, str(color_assignments)]
    # ]
    # save_to_excel('/home/rafael/Documents/HappySet/MIHS/Resultados/Resultado_BC_HappySet.xlsx', result_data, headers=['Instance', 'Model', 'Vertices', 'Edges', 'Time', 'Optimal Value', 'Color Assignments'])

    return cpx.solution.get_objective_value(), valores, tempo_execucao, tempo_execucao_prob_sep, num_user_cuts, cpx.solution.progress.get_num_nodes_processed(),mip_gap,mhis_callback.lp




# Caminho do arquivo de entrada
# arquivo = 'barabasi_albert_n100_q2_k5_colored_2.txt'
arquivo = '/home/rafael/Documents/HappySet/MIHS/inputs/barabasi_albert/100/barabasi_albert_n100_q3_k5_colored_2.txt'
arquivo = "/home/rafael/Documents/HappySet/MIHS/inputs/happygen/output/testes/teste_k2_p0.2_v10.txt"
instancia = GraphInstance(arquivo)
adj_list, precolored, vertices, arestas_list, k = instancia.get_data()

#Chama o modelo com os dados da instância
modelo_MHIS_BC(vertices, arestas=arestas_list, precoloridos=precolored)