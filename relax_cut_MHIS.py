import cplex
import time
import pandas as pd
from readershappyinstances import GraphInstance
import networkx as nx
import numpy as np
import cplex
import openpyxl
import os
import logging

def get_files_in_folder(folder_path, extension=".txt"):
    """
    Retorna uma lista de arquivos com a extensão especificada dentro de uma pasta,
    sem usar o glob.

    :param folder_path: Caminho para a pasta onde os arquivos estão localizados.
    :param extension: Extensão dos arquivos a serem recuperados (padrão é '.txt').
    :return: Lista de arquivos com a extensão fornecida.
    """
    # Verifica se o diretório existe
    if not os.path.isdir(folder_path):
        raise ValueError(f"A pasta '{folder_path}' não existe.")
    
    # Lista os arquivos no diretório
    arquivos = [f for f in os.listdir(folder_path) if f.endswith(extension)]
    
    # Cria o caminho completo de cada arquivo
    arquivos_completos = [os.path.join(folder_path, arquivo) for arquivo in arquivos]
    
    return arquivos_completos


def save_to_excel(output_file, data, headers=None):
    """Salva dados em um arquivo Excel, criando o arquivo e diretórios se necessário."""
    directory = os.path.dirname(output_file)
    if directory and not os.path.exists(directory):
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


def find_RL(instancia, caminho_excel = '/home/rafael/Documents/HappySet/MIHS/Resultados/Resultado_BC_HappySet_december.xlsx'):
    # Carregar o arquivo Excel
    df = pd.read_excel(caminho_excel)
    
    # Procurar o nome da instância na coluna A e obter o valor na coluna J (LP)
    linha = df[df.iloc[:, 0] == instancia]
    
    # Se a instância for encontrada, retornar o valor da coluna J (LP)
    if not linha.empty:
        RL_value = linha.iloc[0, 9]  # Coluna J é a décima (índice 9)
        return RL_value
    else:
        return "Valor da Relaxação da Instância não encontrado."

class RelaxAndCutMHIS:
    def __init__(self, filename):
        self.filename = filename
        self.instance = GraphInstance(filename)
        self.adj_list, self.precolored, self.vertices, self.edges, self.k = self.instance.get_data()
        self.model = cplex.Cplex()
        self.var_h = {}
        self.lambda_multipliers = {}  # Multiplicadores de Lagrange
        self.h_i_to_paths = {i: set() for i in self.vertices}
        self.CA = set()  # Conjunto de restrições atualmente violadas
        self.PA = set()  # Conjunto de restrições previamente ativas
        # self.CI = set()  # Conjunto de restrições inativas
        self.constraints_life = {}  # Vida útil das restrições em PA
        self.max_life = 3  # Parâmetro de extensão de vida
        self._initialize_variables()
        self._initialize_constraints()
        self.model.objective.set_sense(self.model.objective.sense.maximize)
        # Parâmetros do solver
        self.model.parameters.threads.set(1)
        self.model.parameters.timelimit.set(600)
        self.model.parameters.preprocessing.presolve.set(0)
    
    def _initialize_variables(self):
        """Inicializa as variáveis de decisão."""
        for i in self.vertices:
            var_name = f"h_{i}"
            self.var_h[i] = len(self.model.variables.get_names())
            # Coeficientes iniciais da função objetivo são 1
            self.model.variables.add(
                types=["B"], obj=[1], lb=[0], ub=[1], names=[var_name]
            )
    
    def _initialize_constraints(self):
        """Adiciona restrições de pré-coloração."""
        for v in self.precolored:
            print("v precolored", v)
            constr_name = f"precolored_{v}"
            self.model.linear_constraints.add(
                lin_expr=[[[self.var_h[v]], [1]]],
                senses=["E"], rhs=[1], names=[constr_name]
            )
    
    def _update_objective_coefficients(self):
        """Atualiza os coeficientes da função objetivo com base nos multiplicadores atuais."""
        for i in self.vertices:
            sum_lambda = sum(
                self.lambda_multipliers[p] for p in self.h_i_to_paths[i] if p in self.lambda_multipliers
            )
            c_i = 1 - sum_lambda
            self.model.objective.set_linear(self.var_h[i], c_i)
            # print(f"Atualizado coeficiente de h_{i}: {c_i}")

    def _calculate_subgradients(self, solution):
        """Calcula os subgradientes para as restrições dualizadas."""
        subgradients = {}
        # Considera apenas as restrições atualmente ativas (PA e CA)
        for p in self.PA.union(self.CA):
            sum_h = sum(solution[self.var_h[i]] for i in p)
            rhs = len(p) - 1
            g_p = rhs - sum_h
            subgradients[p] = g_p
        # print(f"Subgradientes calculados: {subgradients}")
        return subgradients
    
    def _update_multipliers(self, subgradients, step_size):
        # print('step_size',step_size)
        """Atualiza os multiplicadores de Lagrange usando o método do subgradiente."""
        for p, g_p in subgradients.items():
            if p not in self.lambda_multipliers:
                self.lambda_multipliers[p] = 0
            old_lambda = self.lambda_multipliers[p]
            self.lambda_multipliers[p] = max(old_lambda - step_size * g_p, 0)
            # print(f"Atualizado multiplicador para caminho {p}: de {old_lambda} para {self.lambda_multipliers[p]}")
    
    def _find_violations(self, solution):
        """Identifica restrições violadas (caminhos)."""
        edge_costs = self._calculate_edge_costs(solution)
        graph = nx.Graph()
        for (u, v), cost in edge_costs.items():
            graph.add_edge(u, v, weight=cost)

        violations = []
        precolored_vertices = list(self.precolored.keys())
        for i, v1 in enumerate(precolored_vertices):
            for v2 in precolored_vertices[i + 1:]:
                if self.precolored[v1] == self.precolored[v2]:
                    continue  # Ignora vértices da mesma cor
                try:
                    path = nx.shortest_path(
                        graph, source=v1, target=v2, weight="weight"
                    )
                    path_cost = sum(
                        graph[u][v]['weight'] for u, v in zip(path[:-1], path[1:])
                    )
                    h_v1 = solution[self.var_h[v1]]
                    h_v2 = solution[self.var_h[v2]]
                    total_cost = path_cost + (1 - h_v1) / 2 + (1 - h_v2) / 2
                    if total_cost < 1 - 1e-6:
                        violations.append(tuple(path))
                        # print(f"Violação encontrada: caminho {path} com custo total {total_cost}")
                except nx.NetworkXNoPath:
                    continue
        return violations
    
    def _update_constraint_sets(self, subgradiente):
        """Atualiza os conjuntos CA, PA e CI."""
        PAtoCA = set()
        CAtoPA = set()
        PAtoCI = set()
        for p in self.PA:
            if subgradiente[p] < 0:
                PAtoCA.add(p)
                # self.PA.remove(p)
                del self.constraints_life[p]
                continue
            if p not in self.constraints_life:
                self.constraints_life[p] = 0
            self.constraints_life[p] += 1
            if self.constraints_life[p] > self.max_life:
                # Move para CI
                # self.CI.add(p)
                # self.PA.remove(p)
                PAtoCI.add(p)
                del self.lambda_multipliers[p]
                del self.constraints_life[p]
                for i in p:
                    self.h_i_to_paths[i].remove(p)
        self.PA.difference_update(PAtoCA)
        self.PA.difference_update(PAtoCI)
        # Adiciona novos caminhos a h_i_to_paths
        for p in self.CA:
            if subgradiente[p] >= 0:
                CAtoPA.add(p)
                # self.CA.remove(p)
                self.constraints_life[p] = 1
            else:
                for i in p:
                    if p in self.h_i_to_paths[i]:
                        break
                    else:
                        for j in p:
                            self.h_i_to_paths[j].add(p)
                        break
        self.CA.difference_update(CAtoPA)
        self.CA.update(PAtoCA)
        self.PA.update(CAtoPA)
        # print(f"Conjuntos atualizados: CA={self.CA}, PA={self.PA}")
    
    def _calculate_edge_costs(self, solution):
        """Calcula os custos das arestas."""
        edge_costs = {}
        for u, v in self.edges:
            h_u = solution[self.var_h[u]]
            h_v = solution[self.var_h[v]]
            edge_costs[(u, v)] = (2 - h_u - h_v) / 2
        # print(f"Custos das arestas: {edge_costs}")
        return edge_costs
    
    def solve(self, max_iterations=10000, alpha_0=100, epsilon=1e-5):
        """Resolve o problema usando Relax-and-Cut com relaxação Lagrangeana."""
        UB = float('inf')  
        LB = find_RL(self.filename)

        start_time = time.time()
        best_iteration = 0

        for iteration in range(max_iterations):
            last_iteration = iteration + 1  # Armazena a iteração atual 
            print(f"\nIteração {iteration + 1}")
            self._update_objective_coefficients()
            self.model.solve()
            solution = self.model.solution.get_values()
            # print('solution', solution)
            # print(self.lambda_multipliers[tuple(p)])
            # for p in self.PA.union(self.CA):
            #     print(len(p))
            # print('passou')
            L = self.model.solution.get_objective_value()
            L += sum(self.lambda_multipliers[tuple(p)] * (len(p)-1) for p in self.PA.union(self.CA)) 
            if L< UB:
                best_iteration = iteration
            UB = min(UB, L)
            print("UB, L, LB",UB, L, LB)
            # print('UB - LB',UB - LB)
            best_solution = solution.copy() # Armazena a melhor solução
            # Calcula subgradientes
            self.CA.update(self._find_violations(solution))
            subgradients = self._calculate_subgradients(solution)
            step_size = alpha_0 / (iteration + 1)
            self._update_constraint_sets(subgradients)
            self._update_multipliers(subgradients, step_size)

            if abs(UB - LB) < epsilon:
                print("Convergência atingida.")
                break

        end_time = time.time()
        total_time = end_time - start_time

        # Exemplo de dados a serem salvos
        instance_name = self.filename
        result_data = [
            [instance_name, len(self.vertices), len(self.edges), total_time, UB, LB, UB - LB, max_iterations,last_iteration, best_iteration, str(best_solution)]
        ]
        # Definir o cabeçalho uma única vez no início do arquivo:
        headers = ['Instance', 'Vertices', 'Edges', 'Time', 'UB', 'LB', 'Gap', 'MaxIterations', 'LastIteration', 'BestIteration','Solution']
        save_to_excel('/home/rafael/Documents/HappySet/MIHS/Resultados/Resultados_Relax_and_Cut.xlsx', result_data, headers=headers)

        return UB


class LagrangianRelaxationMHIS:
    def __init__(self, filename):
        """
        Inicializa a instância do problema MHIS com Relaxação Lagrangeana.

        Parâmetros:
        - filename (str): Nome da instância.
        - colors (list): Lista de cores disponíveis (representadas como inteiros, por exemplo, [1, 2, 3]).
        - caminho_excel (str): Caminho para o arquivo Excel contendo os valores da relaxação linear.
        """
        self.filename = filename
        self.instance = GraphInstance(filename)
        self.adj_list, self.precolored, self.V, self.E, self.k = self.instance.get_data()
        self.graph = nx.Graph()
        self.graph.add_edges_from(self.E)
        self.V_prime = set(self.precolored.keys())
        self.P = self._generate_paths()
        self.lambda_multipliers = {tuple(p): 0.0 for p in self.P}

    def _generate_paths(self):
        """
        Gera todos os caminhos mais curtos entre pares de vértices pré-coloridos com cores diferentes.
        """
        P = []
        precolored_vertices = list(self.precolored.keys())
        for i in range(len(precolored_vertices)):
            v1 = precolored_vertices[i]
            color1 = self.precolored[v1]
            for j in range(i + 1, len(precolored_vertices)):
                v2 = precolored_vertices[j]
                color2 = self.precolored[v2]
                if color1 != color2:
                    try:
                        path = nx.shortest_path(self.graph, source=v1, target=v2)
                        P.append(path)
                    except nx.NetworkXNoPath:
                        continue
        return P

    def find_RL(self, instancia, caminho_excel='/home/rafael/Documents/HappySet/MIHS/Resultados/Resultado_BC_HappySet_december.xlsx'):
        """
        Captura o valor da relaxação linear (LP) a partir de um arquivo Excel.

        Parâmetros:
        - instancia (str): Nome da instância a ser buscada.
        - caminho_excel (str): Caminho para o arquivo Excel.

        Retorna:
        - float: Valor da relaxação linear.
        - str: Mensagem de erro se a instância não for encontrada.
        """
        # Carregar o arquivo Excel
        try:
            df = pd.read_excel(caminho_excel)
        except FileNotFoundError:
            return f"Arquivo Excel não encontrado no caminho: {caminho_excel}"

        # Procurar o nome da instância na primeira coluna e obter o valor na décima coluna (J)
        linha = df[df.iloc[:, 0] == instancia]

        # Se a instância for encontrada, retornar o valor da coluna J (índice 9)
        if not linha.empty:
            RL_value = linha.iloc[0, 9]  # Coluna J é a décima (índice 9)
            try:
                RL_float = float(RL_value)
                print(f"Relaxação Linear (RL) para a instância '{instancia}': {RL_float}")
                return RL_float
            except ValueError:
                return f"Valor da Relaxação Linear não é numérico para a instância '{instancia}'."
        else:
            return f"Valor da Relaxação da Instância '{instancia}' não encontrado."

    def _compute_sum_lambda_p(self):
        """
        Calcula a soma dos multiplicadores lambda_p para cada vértice i.
        """
        sum_lambda_p = {i: 0.0 for i in self.V}
        for p in self.P:
            lambda_p = self.lambda_multipliers[tuple(p)]
            for i in p:
                sum_lambda_p[i] += lambda_p
        return sum_lambda_p

    def _solve_subproblem(self, sum_lambda_p):
        """
        Resolve o subproblema Lagrangeano, determinando h_i.

        Parâmetros:
        - sum_lambda_p (dict): Soma dos multiplicadores para cada vértice.

        Retorna:
        - h (dict): Decisão de felicidade para cada vértice.
        """
        h = {}
        for i in self.V:
            if i in self.V_prime:
                # Vértices pré-coloridos devem ser felizes
                h[i] = 1
            else:
                h[i] = 1 if sum_lambda_p[i] <= 1.0 else 0
        return h

    def _compute_subgradients(self, h):
        """
        Calcula os subgradientes para cada caminho p em P.

        Parâmetros:
        - h (dict): Decisão de felicidade para cada vértice.

        Retorna:
        - subgradients (dict): Subgradiente para cada caminho p.
        """
        subgradients = {}
        for p in self.P:
            sum_h = sum(h[i] for i in p)
            rhs = len(p) - 1
            g_p = sum_h - rhs
            subgradients[tuple(p)] = g_p
        return subgradients

    def _compute_L(self, h):
        """
        Calcula o valor da função objetivo Lagrangeana L(λ).

        Parâmetros:
        - h (dict): Decisão de felicidade para cada vértice.

        Retorna:
        - L (float): Valor da função objetivo Lagrangeana.
        """
        # L = sum(h_i) + sum(lambda_p * (|p| -1))
        L = 0
        for i in self.V:
            if h[i] >0.5:
                L += 1 - sum(self.lambda_multipliers[tuple(p)] for p in self.P  if i in p)
        L += sum(self.lambda_multipliers[tuple(p)] * (len(p)-1 ) for p in self.P)
        return L

        # L =  sum(1 - sum(self.lambda_multipliers[tuple(p)] for p in self.P if i in p)* h[i]) + sum(self.lambda_multipliers[tuple(p)] * (len(p)-1 ) for p in self.P)
        # return L

    def solve(self, max_iterations=10, alpha_0=2.0, epsilon=1e-5, tolerance=1e-6):
        """
        Resolve o problema dual de Lagrange usando o método do subgradiente.

        Parâmetros:
        - max_iterations (int): Número máximo de iterações.
        - alpha_0 (float): Fator inicial para o passo.
        - epsilon (float): Tolerância para a convergência.
        - tolerance (float): Tolerância para reduzir pi.

        Retorna:
        - best_h (dict): Melhor solução encontrada para h_i.
        """
        # Inicializa Lower Bound (LB) usando a relaxação linear pré-computada
        LB = self.find_RL(self.filename)
        if isinstance(LB, str):
            print(LB)
            return None

        # Inicializa Upper Bound (UB) como o máximo possível de vértices felizes
        UB = len(self.V)
        best_h = None

        for iteration in range(1, max_iterations +1 ):
            print(f"\n--- Iteração {iteration} ---")

            # 1. Computar soma dos multiplicadores para cada vértice
            sum_lambda_p = self._compute_sum_lambda_p()
            print('Soma dos multiplicadores para cada vértice (sum_lambda_p):', sum_lambda_p)

            # 2. Resolver o subproblema Lagrangeano
            h = self._solve_subproblem(sum_lambda_p)
            print('Decisão de felicidade (h):', h)

            # 3. Calcular L(λ)
            L = self._compute_L(h)
            print('Valor da função objetivo Lagrangeana (L):', L)

            # 4. Atualizar o Upper Bound (UB) se necessário
            if L < UB:
                UB = L  # Atualiza o Upper Bound com o novo valor de L
                best_h = h.copy()
                print(f"Novo UB atualizado: {UB}")
            else:
                print("UB não foi atualizado.")

            # 5. Calcular os subgradientes
            subgradients = self._compute_subgradients(h)
            print('Subgradientes:', subgradients)

            # 6. Calcular a norma do subgradiente ao quadrado
            g_norm_sq = sum(g_p ** 2 for g_p in subgradients.values())
            print('Norma do subgradiente ao quadrado:', g_norm_sq)

            if g_norm_sq == 0:
                print("Norma do subgradiente é zero. Convergência atingida.")
                break

            # 7. Calcular o tamanho do passo (step_size) com passo decrescente
            duality_gap = UB - LB
            if duality_gap <=0:
                print(f"Lacuna de dualidade {duality_gap} é menor ou igual a zero. Parando.")
                break

            # Implementando um passo decrescente para evitar overshooting
            step_size = alpha_0 / (iteration ** 0.5)
            print('Tamanho do passo (step_size):', step_size)

            # 8. Atualizar os multiplicadores λ_p na direção oposta do subgradiente
            for p in self.P:
                g_p = subgradients[tuple(p)]
                lambda_p = self.lambda_multipliers[tuple(p)]
                lambda_p_new = max(0.0, lambda_p - step_size * g_p)
                self.lambda_multipliers[tuple(p)] = lambda_p_new
                print(f"λ_p para caminho {p} atualizado para: {lambda_p_new}")

            # 9. Verificar convergência com base na lacuna de dualidade
            if duality_gap < epsilon:
                print(f"Lacuna de dualidade {duality_gap} é menor que a tolerância {epsilon}. Parando.")
                break

            # 10. Exibir o progresso da iteração
            print(f"Iteração {iteration}: LB = {LB}, UB = {UB}, Gap = {UB - LB}")

        print("\n--- Resultados Finais ---")
        print(f"Lower Bound (LB): {LB}")
        print(f"Upper Bound (UB): {UB}")
        if best_h is not None:
            print(f"Número de vértices felizes: {sum(best_h.values())}")
            print("Vértices felizes:")
            for v in self.V:
                if best_h[v] == 1:
                    print(f"Vértice {v} está na subgrafo induzida e é feliz.")
        else:
            print("Nenhuma solução viável encontrada.")
        return best_h


# def heuristica_mhis(adj_list, precolored, vertices):
#     # Passo 1: Propagar cores de vértices pré-coloridos
#     for c in set(precolored.values()):
#         for v in vertices:
#             if precolored.get(v) == c:
#                 for w in adj_list[v]:
#                     if w not in precolored:
#                         pode_colorear = True
#                         for vizinho_w in adj_list[w]:
#                             if vizinho_w in precolored and precolored[vizinho_w] != c:
#                                 pode_colorear = False
#                                 break
#                         if pode_colorear:
#                             precolored[w] = c

#     # Passo 2: Atribuir a primeira cor para vértices não coloridos cujos vizinhos também não são coloridos
#     for v in vertices:
#         if v not in precolored:
#             todos_vizinhos_descoloridos = True
#             for w in adj_list[v]:
#                 if w in precolored:
#                     todos_vizinhos_descoloridos = False
#                     break
#             if todos_vizinhos_descoloridos:
#                 precolored[v] = 1  # Atribuir a primeira cor (ou qualquer cor válida)

#     # Passo 3: Contar os vértices felizes
#     num_felizes = 0
#     felizes = []
#     for v in vertices:
#         if v in precolored:  
#             cor_v = precolored[v]
#             todos_vizinhos_felizes = True
#             for w in adj_list[v]:
#                 if w in precolored:
#                     if precolored[w] != cor_v:
#                         todos_vizinhos_felizes = False
#                     break
#                 else:
#                     todos_vizinhos_felizes = False
#             if todos_vizinhos_felizes:
#                 felizes.append(v)
#                 num_felizes += 1

#     return precolored, num_felizes, felizes




# Exemplo de uso

if __name__ == "__main__":
    filename = '/home/rafael/Documents/HappySet/MIHS/inputs/happygen/output/testes/7-10/graph_scale_free_k2_p0.4_v10_20241125_165041.txt'
    instance = GraphInstance(filename)
    folder_path = '/home/rafael/Documents/HappySet/MIHS/inputs/happygen/output/testes/testes_cp_benders'
    folder_path = '/home/rafael/Documents/HappySet/MIHS/inputs/happygen/output/testes/7-10'
# folder_path = '/home/rafael/Documents/HappySet/MIHS/inputs/happygen/output/testes/20-50-100 (reduce)'
    arquivos = get_files_in_folder(folder_path)

    for arquivo in arquivos:
        print("\n Resolvendo RelaxAndCutMHIS:")
        solver = RelaxAndCutMHIS(arquivo)
        best_value = solver.solve()
        print(f"Valor ótimo: {best_value}")

    # print("\n Resolvendo a RELAXAÇÃO LAGRANGEANA, SEM OS CORTES:")
    # adj_list, precolored, vertices, edges, k = instance.get_data()

    # print("Lista de Adjacências:")
    # print(adj_list)
    # print("\nVértices pré-coloridos:")
    # print(precolored)

    # solver = LagrangianRelaxationMHIS(filename)
    # best_h = solver.solve()

    # # Chamar a função
    # RL = find_RL(filename)

    # # Imprimir o resultado
    # print(f"O valor de LP para a instância é: {RL}")



