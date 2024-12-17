from math import fabs
import sys
import cplex
from cplex.callbacks import UserCutCallback, LazyConstraintCallback
from cplex.exceptions import CplexError
from readershappyinstances import GraphInstance
from docplex.mp.model import Model
import os
import time
import traceback
import networkx as nx

###################################################################### BENDERS AUTO ######################################################################

def create_log_folder():
    log_folder = '/home/rafael/Downloads/Bendersauto'
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)  # Cria a pasta se não existir
    return log_folder

def model_benders(lista_viz, k):
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
        mdl.add_constraint(h[i] >= mdl.sum(y[j] for j in lista_viz[i]) - len(lista_viz[i]) + y[i])

    # Definindo parâmetro para setar o Benders automáitco (sem criar anotação)
    mdl.parameters.benders.strategy = 3
    mdl.parameters.mip.strategy.heuristicfreq.set(-1)
    mdl.parameters.mip.cuts.mircut.set(-1)
    mdl.parameters.mip.cuts.implied.set(-1)
    mdl.parameters.mip.cuts.gomory.set(-1)
    mdl.parameters.mip.cuts.flowcovers.set(-1)
    mdl.parameters.mip.cuts.pathcut.set(-1)
    mdl.parameters.mip.cuts.liftproj.set(-1)
    mdl.parameters.mip.cuts.zerohalfcut.set(-1)
    mdl.parameters.mip.cuts.cliques.set(-1)
    mdl.parameters.mip.cuts.covers.set(-1)
    mdl.parameters.threads.set(1)
    mdl.parameters.clocktype.set(1)
    mdl.parameters.timelimit.set(60)
    mdl.parameters.preprocessing.presolve.set(0)
    mdl.parameters.preprocessing.boundstrength.set(0)
    mdl.parameters.preprocessing.coeffreduce.set(0)
    mdl.parameters.preprocessing.relax.set(0)
    mdl.parameters.preprocessing.aggregator.set(0)
    mdl.parameters.preprocessing.reduce.set(0)
    mdl.parameters.preprocessing.reformulations.set(0)

    # # Criando o logger para capturar as saídas no log
    # logger = setup_logger('OriginalModel', log_file)
    # with redirect_output(log_file):
    #     logger.info("Iniciando a otimização com o modelo original.")

    start_time = time.time()  # Tempo de início da execução
    solution = mdl.solve()
    exec_time = time.time() - start_time  # Tempo de execução

    if solution:
        selected_vertices = [i for i in range(n) if solution.get_value(y[i]) == 1]
        happy_vertices = [i for i in range(n) if solution.get_value(h[i]) == 1]
        objective_value = solution.objective_value  # Valor da função objetivo
        print(f"Vértices Selecionados: {selected_vertices}")
        print(f"Vértices Felizes: {happy_vertices}")
        print(f"Valor da solução: {objective_value}")
        return selected_vertices, happy_vertices, exec_time, objective_value
    else:
        print("Solução não encontrada!")
        return None, None, exec_time, None

###################################################################### BENDERS MANUAL ######################################################################

class WorkerLP():
    """Classe WorkerLP para o MaxHS, responsável por resolver o subproblema
    e gerar cortes de Benders a partir da solução atual do mestre.
    """

    def __init__(self, adj_list):
        self.adj_list = adj_list
        self.n = len(adj_list)
        self.model = cplex.Cplex()
        self._setup_model()
        self.cut_lhs = None
        self.cut_rhs = None

    def _setup_model(self):
        """Configura o modelo do subproblema MaxHS. Variáveis h_i tratadas como contínuas e restrições adicionadas dinamicamente no método 'separate'.
        """
        self.model.set_results_stream(None)
        self.model.set_log_stream(None)

        self.model.set_problem_type(cplex.Cplex.problem_type.LP)

        self.model.objective.set_sense(self.model.objective.sense.maximize)
        self.h_vars = [f"h_{i}" for i in range(self.n)]
        self.model.variables.add(names=self.h_vars, lb=[0.0]*self.n, ub=[1.0]*self.n)

        # Função objetivo: somatório de h_i
        objective_coeffs = [1.0] * self.n  
        self.model.objective.set_linear(list(zip(self.h_vars, objective_coeffs)))


        # Restrições: h_i <= y_i
        for i in range(self.n):
            self.model.linear_constraints.add(
                lin_expr=[cplex.SparsePair(ind=[self.h_vars[i]], val=[1.0])],
                senses=["L"],
                rhs=[1.0],
                names=[f"h_{i}_leq"]
            )

        
        # Restrições: h_i <= y_j for j in Gamma(i)
        for i in range(self.n):
            for j in self.adj_list[i]:
                self.model.linear_constraints.add(
                    lin_expr=[cplex.SparsePair(ind=[self.h_vars[i]], val=[1.0])],
                    senses=["L"],
                    rhs=[0.0],
                    names=[f"h_{i}_leq_{j}"]
                )
            # print("vertice", i)

        # # h_i <= y_i + sum_{j in Gamma(i)} y_j
        for i in range(self.n):
        #     # print("vertice", i)
            rhs = 0 + sum(0 for j in self.adj_list[i])
            # print('yi',y_solution[i])
            # print('rhs',rhs)
            self.model.linear_constraints.add(
                lin_expr=[cplex.SparsePair(ind=[self.h_vars[i]], val=[1.0])],
                senses=["G"],
                rhs=[rhs],
                names=[f"h_{i}_geq"]
            )

    def separate(self, y_solution, y_vars, eta_value):
        """
        Resolve o subproblema para uma dada solução \bar{y} do Mestre,
        extrai os valores duais associados a cada restrição,
        verifica violação de corte (eta > eta_value) e, se violado, constrói o corte de Benders.
        """

        # Atualizar lado direito das restrições
        for i in range(self.n):
            self.model.linear_constraints.set_rhs([(f"h_{i}_leq", y_solution[i])])

        # h_i <= y_j
        for i in range(self.n):
            for j in self.adj_list[i]:
                self.model.linear_constraints.set_rhs([(f"h_{i}_leq_{j}", y_solution[j])])

        # h_i >= y_i + sum_{j in Gamma(i)} y_j - |Gamma(i)|
        for i in range(self.n):
            rhs = y_solution[i] + sum(y_solution[j] for j in self.adj_list[i]) - len(self.adj_list[i])
            self.model.linear_constraints.set_rhs([(f"h_{i}_geq", rhs)])

        # Resolver o subproblema
        self.model.write("subproblem_new.lp")  # Opcional: Salvar o modelo do subproblema para depuração
        self.model.solve()
        status = self.model.solution.get_status()

        if status == self.model.solution.status.optimal or status == 101:
            eta = self.model.solution.get_objective_value()
            duals = self.model.solution.get_dual_values()

            # Separar os duals
            lambda_duals = duals[:self.n]  # Duals das restrições h_i <= y_i
            m = sum(len(self.adj_list[i]) for i in range(self.n))
            alpha_duals = duals[self.n:self.n + m]  # Duals das restrições h_i <= y_j
            gamma_duals = duals[self.n + m:]  # Duals das restrições h_i >= y_i + sum y_j - |Γ(i)|

            # Organizar alpha_duals por i
            alpha_duals_per_i = []
            current = 0
            for i in range(self.n):
                m_i = len(self.adj_list[i])
                if m_i > 0:
                    alpha_duals_per_i.append(alpha_duals[current: current + m_i])
                    current += m_i
                else:
                    alpha_duals_per_i.append([])

            # Construir os coeficientes do corte conforme a fórmula corrigida
            coef_dict = {var: 0.0 for var in y_vars}  # Inicializar coeficientes para cada y_i

            for i in range(self.n):
                # Coeficiente para y_i: lambda_i - gamma_i
                coef_dict[y_vars[i]] += lambda_duals[i] - gamma_duals[i]

                # Coeficientes para y_j: alpha_ij - gamma_i para cada j ∈ Γ(i)
                for j_idx, j in enumerate(self.adj_list[i]):
                    alpha_ij = alpha_duals_per_i[i][j_idx]
                    coef_dict[y_vars[j]] += alpha_ij - gamma_duals[i]

            # Calcular o lado direito do corte: eta - sum_i |Gamma(i)| gamma_i
            sum_gamma = sum(len(self.adj_list[i]) * gamma_duals[i] for i in range(self.n))
            cut_rhs = eta - sum_gamma

            # Construir a lista de coeficientes alinhada com y_vars
            cut_coefs = [coef_dict[y] for y in y_vars]

            # Construir o corte
            self.cut_lhs = cplex.SparsePair(ind=y_vars, val=cut_coefs)
            self.cut_rhs = cut_rhs

            # Depuração: imprimir os coeficientes e rhs
            print("lambda_duals:", lambda_duals)
            print("alpha_duals_per_i:", alpha_duals_per_i)
            print("gamma_duals:", gamma_duals)
            print("coef_dict:", coef_dict)
            print("cut_coefs:", cut_coefs)
            print("cut_rhs:", cut_rhs)

            # Verificar se há violação do corte (C < eta)
            if eta < self.cut_rhs - 1e-5:
                return True
            else:
                return False



class MaxHSCallback():
    """Callback para o problema MaxHS.
    Separ cortes de Benders em soluções fracionárias (user cuts) e soluções inteiras (lazy constraints) usando a função invoke e criação de WorkerLP.
    """

    def __init__(self, num_threads, num_nodes, y_vars, eta_var, adj_list):
        self.num_threads = num_threads
        self.num_nodes = num_nodes
        self.y_vars = y_vars
        self.adj_list = adj_list
        self.eta_var = eta_var
        self.worker = WorkerLP(self.adj_list)
        self.k = None

    def separate_user_cuts(self, context):
        """Separar cortes em soluções fracionárias (user cuts)."""
        # Obter solução fracionária
        sol = []
        for i in range(self.num_nodes):
            sol.append(context.get_relaxation_point(self.y_vars[i]))
        eta_value = context.get_relaxation_point(self.eta_var)
        print('eta_value master',eta_value)
        print('user cuts sol',sol)

        # Separar cortes
        if self.worker.separate(sol, self.y_vars, eta_value):
            context.add_user_cut(cut=self.worker.cut_lhs, sense='L', rhs=self.worker.cut_rhs, local=False)
            print("Corte de Benders (UserCut) adicionado.")

    def separate_lazy_constraints(self, context):
        """Separar cortes em soluções inteiras (lazy constraints)."""
        if not context.is_candidate_point():
            raise Exception('Solução não limitada detectada.')
        
        sol = []
        for i in range(self.num_nodes):
            sol.append(context.get_candidate_point(self.y_vars[i]))
        eta_value = context.get_candidate_point(self.eta_var)
        print('eta_value master',eta_value)
        print('lazy cuts sol',sol)

        if self.worker.separate(sol, self.y_vars, eta_value):
            print("Worker Separate - reject_candidate")
            context.reject_candidate(constraints=[self.worker.cut_lhs], senses='L', rhs=[self.worker.cut_rhs])
            print("Corte de Benders (LazyConstraint) adicionado.")

    def invoke(self, context):
        """Método chamado pelo CPLEX durante a otimização."""
        try:
            # thread_id = context.get_int_info(cplex.callbacks.Context.info.thread_id)
            cid = context.get_id()
            # if cid == cplex.callbacks.Context.id.thread_up:
            #     print('cplex.callbacks.Context.id.thread_up',cplex.callbacks.Context.id.thread_up)
            #     self.workers[thread_id] = WorkerLP(self.adj_list)
            # elif cid == cplex.callbacks.Context.id.thread_down:
            #     self.workers[thread_id] = None
            #     print('cplex.callbacks.Context.id.thread_down',cplex.callbacks.Context.id.thread_down)
            if cid == cplex.callbacks.Context.id.relaxation:
                print("User Cut Callback - Id Relaxation")
                self.separate_user_cuts(context)
                # print('self.separate_user_cuts(context, self.workers[thread_id])',self.separate_user_cuts(context, self.workers[thread_id]))
            elif cid == cplex.callbacks.Context.id.candidate:
                print("Lazy Cut Callback - Id Candidate")
                # print("thread_id",thread_id)
                self.separate_lazy_constraints(context)
                # print('self.separate_lazy_constraints(context, self.workers[thread_id])',self.separate_lazy_constraints(context, self.workers[thread_id]))
            else:
                print(f"Contexto inesperado: {cid}")
        except:
            print("HelloExcept")
            info = sys.exc_info()
            print('#### Exception in callback:', info[0])
            print('####                       ', info[1])
            traceback.print_tb(info[2], file=sys.stdout)
            raise


def configure_master(adj_list, k):
    """Configura o modelo mestre do MaxHS."""
    master_model = cplex.Cplex()
    n = len(adj_list)
    y_vars = [f"y_{i}" for i in range(n)]
    eta_var = "eta"
    
    master_model.variables.add(names=y_vars, lb=[0]*n, ub=[1]*n, types=["B"]*n) # Adicionar variáveis y_i binárias
    master_model.variables.add(names=[eta_var], lb=[0.0], ub=[n]) # Adicionar variável eta contínua, UB natural é n, que nao há como selecionar mais do que n vértices
    master_model.objective.set_sense(master_model.objective.sense.maximize) # Objetivo: maximizar eta
    master_model.objective.set_linear([(eta_var, 1.0)] + [(var, 0.0) for var in y_vars])

    # Restrição sum y_i = k
    master_model.linear_constraints.add(
        lin_expr=[cplex.SparsePair(ind=y_vars, val=[1.0]*n)],
        senses=["E"],
        rhs=[k],
        names=["Cardinality"]
    )
    return master_model, y_vars, eta_var


def process_instance(instance_file, k_value):
    """
    Processa a instância do MaxHS, configurando o mestre, callback e resolvendo.
    """
    # Ler a instância usando GraphInstance
    instancia = GraphInstance(instance_file)
    adj_list, _, _, _, k = instancia.get_data()
    n = len(adj_list)
    # print("self.adj_list",adj_list)
    
    master_model, y_vars, eta_var = configure_master(adj_list, k_value)
    num_threads = 1 #master_model.get_num_cores()
    master_model.parameters.threads.set(num_threads)
    
    # Criando callback
    maxhs_cb = MaxHSCallback(num_threads, n, y_vars, eta_var, adj_list)
    maxhs_cb.adj_list = adj_list  
    
    # Definir contextos a serem tratados: relaxation, candidate, thread_up, thread_down
    #contextmask = cplex.callbacks.Context.id.thread_up
    #contextmask |= cplex.callbacks.Context.id.thread_down
    contextmask = cplex.callbacks.Context.id.candidate
    # separar também cortes fracionários:
    contextmask |= cplex.callbacks.Context.id.relaxation
    master_model.set_callback(maxhs_cb, contextmask)

    master_model.parameters.mip.display.set(1)
    master_model.parameters.timelimit.set(60)
    master_model.parameters.preprocessing.presolve.set(master_model.parameters.preprocessing.presolve.values.off)
    master_model.parameters.mip.strategy.search.set(master_model.parameters.mip.strategy.search.values.traditional)
    master_model.parameters.mip.strategy.heuristicfreq.set(-1)
    master_model.parameters.mip.cuts.mircut.set(-1)
    master_model.parameters.mip.cuts.implied.set(-1)
    master_model.parameters.mip.cuts.gomory.set(-1)
    master_model.parameters.mip.cuts.flowcovers.set(-1)
    master_model.parameters.mip.cuts.pathcut.set(-1)
    master_model.parameters.mip.cuts.liftproj.set(-1)
    master_model.parameters.mip.cuts.zerohalfcut.set(-1)
    master_model.parameters.mip.cuts.cliques.set(-1)
    master_model.parameters.mip.cuts.covers.set(-1)
    master_model.parameters.threads.set(1)
    master_model.parameters.clocktype.set(1)
    master_model.parameters.preprocessing.presolve.set(0)
    master_model.parameters.preprocessing.boundstrength.set(0)
    master_model.parameters.preprocessing.coeffreduce.set(0)
    master_model.parameters.preprocessing.relax.set(0)
    master_model.parameters.preprocessing.aggregator.set(0)
    master_model.parameters.preprocessing.reduce.set(0)
    master_model.parameters.preprocessing.reformulations.set(0)

    start_time = time.time()
    master_model.write("master.lp")
    master_model.solve()
    exec_time = time.time() - start_time

    solution = master_model.solution
    status = solution.get_status()

    if status == solution.status.MIP_optimal or status == 101:
        y_values = solution.get_values(y_vars)
        eta_value = solution.get_values(eta_var)
        selected_vertices = [i for i, val in enumerate(y_values) if val > 0.5]
        print(f"\nConjunto Máximo Feliz S*: {selected_vertices}")
        print(f"Número Máximo de Vértices Felizes: {eta_value}")
        print(f"Tempo de execução: {exec_time:.2f} segundos")
    else:
        print(f"\nSolução não ótima. Status: {solution.get_status_string()}")
        print(f"Tempo de execução: {exec_time:.2f} segundos")


def main():
    if len(sys.argv) < 3:
        print("Instrução de uso - para rodar siga o formato: python maxhs_benders.py <k> <instance_file>")
        sys.exit(-1)
    k_value = int(sys.argv[1])
    instance_file = sys.argv[2]
    process_instance(instance_file, k_value)

if __name__ == "__main__":
    main()
















































