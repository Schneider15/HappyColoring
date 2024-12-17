# import cplex
# from readershappyinstances import GraphInstance
# import readershappyinstances
# import networkx as nx
# import time
# from docplex.cp.model import CpoModel

# def maxhs_cp(lista_viz, k):
#     mdl = CpoModel()
#     n = len(lista_viz)  
#     y = mdl.binary_var_list(n, name="y")  
#     h = mdl.binary_var_list(n, name="h")  

#     mdl.maximize(mdl.sum(h[i] for i in range(n)))
#     mdl.add_constraint(mdl.sum(y) == k)
#     for i in range(n):
#         mdl.add_constraint(h[i] <= y[i])
#     for i in range(n):
#         for j in lista_viz[i]:  # j ∈ Γ(i)
#             mdl.add_constraint(h[i] <= y[j])
#     for i in range(n):
#         # h_i >= sum(y_j for j in Γ(i)) - |Γ(i)| + y_i
#         mdl.add_constraint(h[i] >= mdl.sum(y[j] for j in lista_viz[i]) - len(lista_viz[i]) + y[i])

#     cpoptimizer_path = '/home/rafael/CPLEX_Studio2211/cpoptimizer/bin/x86-64_linux/cpoptimizer'
#     solution = mdl.solve(execfile=cpoptimizer_path)
#     # solution = mdl.solve()

#     # Se a solução for encontrada, retorna os vértices selecionados
#     if solution:
#         selected_vertices = [i for i in range(n) if solution.get_value(y[i]) == 1]
#         happy_vertices = [i for i in range(n) if solution.get_value(h[i]) == 1]
#         return selected_vertices, happy_vertices
#     else:
#         print("Solução não encontrada!")
#         return None, None

# # Exemplo de uso (adapte conforme sua estrutura de dados)
# # Carregar a instância do problema
# arquivo = '/home/rafael/Documents/HappySet/MIHS/inputs/happygen/output/testes/teste_k2_p0.2_v10.txt'
# instancia = GraphInstance(arquivo)

# # Obter os dados
# adj_list, precolored, vertices, arestas, k = instancia.get_data()

# # Imprimir os dados carregados
# print("Lista de Adjacências:")
# print(adj_list)

# print("\nVértices pré-coloridos:")
# print(precolored)

# print("\nVértices:")
# print(vertices)

# print("\nArestas:")
# print(arestas)

# # print("\nValor de k:")
# # print(k)

# # Resolver o problema
# for k in range(11):
#     print("############################################################################# Iniciando execução #############################################################################")
#     selected_vertices, happy_vertices = maxhs_cp(adj_list, k)

#     # Exibir o resultado
#     if selected_vertices is not None:
#         print("Vértices selecionados:", selected_vertices)
#         print("Vértices felizes:", happy_vertices)
#         print("Valor de k considerado", k)
#         print("Número de felizes", len(happy_vertices))

import cplex
from readershappyinstances import GraphInstance
import readershappyinstances
import networkx as nx
import time
from docplex.mp.model import Model


def maxhs_cp(lista_viz, k):
    mdl = Model(log_output=True)
    n = len(lista_viz)  
    y = mdl.binary_var_list(n, name="y")  
    h = mdl.continuous_var_list(n, name="h")  

    mdl.maximize(mdl.sum(h[i] for i in range(n)))
    mdl.add_constraint(mdl.sum(y) == k)
    for i in range(n):
        mdl.add_constraint(h[i] <= y[i])
    for i in range(n):
        for j in lista_viz[i]:  # j ∈ Γ(i)
            mdl.add_constraint(h[i] <= y[j])
    for i in range(n):
        # h_i >= sum(y_j for j in Γ(i)) - |Γ(i)| + y_i
        mdl.add_constraint(h[i] >= mdl.sum(y[j] for j in lista_viz[i]) - len(lista_viz[i]) + y[i])

    # mdl.parameters.benders.strategy.set(mdl.parameters.benders.strategy.values.full)
    cpoptimizer_path = '/home/rafael/CPLEX_Studio2211/cpoptimizer/bin/x86-64_linux/cpoptimizer'
    solution = mdl.solve(execfile=cpoptimizer_path)
    # solution = mdl.solve()

    # Se a solução for encontrada, retorna os vértices selecionados
    if solution:
        selected_vertices = [i for i in range(n) if solution.get_value(y[i]) == 1]
        happy_vertices = [i for i in range(n) if solution.get_value(h[i]) == 1]
        return selected_vertices, happy_vertices
    else:
        print("Solução não encontrada!")
        return None, None

# Exemplo de uso, carregar instância do problema
arquivo = '/home/rafael/Documents/HappySet/MIHS/inputs/happygen/output/testes/teste_k2_p0.2_v10.txt'
instancia = GraphInstance(arquivo)
adj_list, precolored, vertices, arestas, k = instancia.get_data()

print("Lista de Adjacências:")
print(adj_list)
print("\nVértices pré-coloridos:")
print(precolored)
print("\nVértices:")
print(vertices)
print("\nArestas:")
print(arestas)

# print("\nValor de k:")
# print(k)

# Resolver o problema
for k in range(11):
    print("############################################################################# Iniciando execução #############################################################################")
    selected_vertices, happy_vertices = maxhs_cp(adj_list, k)

    # Exibir o resultado
    if selected_vertices is not None:
        print("Vértices selecionados:", selected_vertices)
        print("Vértices felizes:", happy_vertices)
        print("Valor de k considerado", k)
        print("Número de felizes", len(happy_vertices))


#TESTES GER COL
# from cplex.callbacks import LazyConstraintCallback, UserCutCallback


# class ColumnGenerationModel:
#     """
#     Modelo de geração de colunas para o problema MAXHV.
#     """
#     def __init__(self, vertices, arestas, precolored, k):
#         self.vertices = vertices
#         self.n = len(vertices)
#         self.arestas = arestas
#         self.precolored = precolored
#         self.k = k
#         self.model = cplex.Cplex()
#         self.master_problem = None
#         self.subproblem = None
    
#     def _initialize_master_problem(self):
#         """
#         Inicializa o problema mestre para o modelo de geração de colunas.
#         """
#         self.master_problem = cplex.Cplex()
#         self.master_problem.objective.set_sense(self.master_problem.objective.sense.maximize)
        
#         # Variáveis de decisão iniciais
#         for v in self.graph.vertices:
#             self.master_problem.variables.add(
#                 obj=[0.0],
#                 names=[f"x_{v}"],
#                 lb=[0.0],
#                 ub=[1.0],
#                 types=["C"]
#             )
        
#         # Restrições de vertice feliz preservado na pré coloração
#         for v in self.graph.precolored_vertices:
#             self.master_problem.linear_constraints.add(
#                 lin_expr=[[f"x_{v}", 1.0]],
#                 senses=["E"],
#                 rhs=[1.0]
#             )
    
#     def _initialize_subproblem(self):
#         """
#         Configura o problema de precificação para identificar colunas violadas.
#         """
#         self.subproblem = cplex.Cplex()
#         self.subproblem.set_problem_type(self.subproblem.problem_type.LP)
#         self.subproblem.objective.set_sense(self.subproblem.objective.sense.minimize)
        
#         # Variáveis e restrições específicas do subproblema
#         # Adicionar lógica baseada na formulação de precificação MAXHV
    
#     def _solve_subproblem(self):
#         """
#         Resolve o subproblema para encontrar colunas violadas.
#         """
#         self.subproblem.solve()
#         if self.subproblem.solution.get_status() == self.subproblem.solution.status.optimal:
#             reduced_cost = self.subproblem.solution.get_objective_value()
#             if reduced_cost < 0:
#                 # Retorna a coluna identificada
#                 return self._extract_column()
#         return None
    
#     def _extract_column(self):
#         """
#         Extrai a coluna identificada no subproblema.
#         """
#         column = {}
#         # Lógica para mapear a solução do subproblema em uma nova coluna para o mestre
#         return column
    
#     def solve(self):
#         """
#         Executa o modelo de geração de colunas até convergência.
#         """
#         self._initialize_master_problem()
#         self._initialize_subproblem()
        
#         while True:
#             self.master_problem.solve()
            
#             # Resolve o subproblema para encontrar novas colunas
#             new_column = self._solve_subproblem()
#             if not new_column:
#                 break
            
#             # Adiciona a nova coluna ao problema mestre
#             self.master_problem.variables.add(
#                 obj=[new_column['cost']],
#                 names=[new_column['name']],
#                 columns=[cplex.SparsePair(new_column['indices'], new_column['values'])]
#             )
        
#         # Retorna a solução final
#         return self.master_problem.solution.get_values()


# if __name__ == "__main__":
#     # Exemplo de uso
#     file_path = ""
#     graph_instance = GraphInstance(file_path)
#     model = ColumnGenerationModel(graph_instance)
#     solution = model.solve()
    
#     print("Solução:", solution)


