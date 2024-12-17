import cplex
from cplex import Cplex, CplexError, SparsePair
import readershappyinstances
from readershappyinstances import GraphInstance
import matplotlib.pyplot as plt
import networkx as nx
import time
            
########################################################### Geração de colunas #####################################################:

class MaxHVColumnGeneration:
    def __init__(self, adj_list, precolored, vertices, k):
        self.adj_list = adj_list
        self.precolored = precolored
        self.vertices = vertices
        self.k = k
        self.dual_values = None

        print(f"Lista de adjacência: {self.adj_list}")
        print(f"Pré-coloridos: {self.precolored}")
        print(f"Vértices: {self.vertices}")
        print(f"Número de cores k: {self.k}")

    def initialize_columns(self):
        """Inicializa as colunas do problema mestre com h_k(r) corretamente definido."""
        columns = []
        col_names = []
        obj = []

        # Colunas para pré-coloridos
        for color in range(1, self.k + 1):
            r = [v for v, c in self.precolored.items() if c == color]
            if not r:
                continue
            col = [0] * len(self.vertices)
            for v in r:
                col[v] = 1
            # Compute h_k(r): número de vértices em r que são felizes
            hkr = 0
            for v in r:
                neighbors = self.adj_list[v]
                all_neighbors_in_r = True
                for j in neighbors:
                    if j not in r or self.precolored.get(j, None) != color:
                        all_neighbors_in_r = False
                        break
                if all_neighbors_in_r:
                    hkr += 1
            obj.append(hkr)
            columns.append(col)
            col_names.append(f"lambda_pre_{color}")

        # Colunas para não pré-coloridos (inicialmente cada vértice sozinho)
        for v in self.vertices:
            if v not in self.precolored:
                col = [0] * len(self.vertices)
                col[v] = 1
                hkr = 0
                if not self.adj_list[v]:  # Vértice sem vizinhos
                    hkr = 1
                obj.append(hkr)
                columns.append(col)
                col_names.append(f"lambda_free_{v}")

        print(f"Colunas iniciais: {col_names}")
        print(f"Objetivo das colunas iniciais: {obj}")
        return columns, col_names, obj

    def solve_master_problem(self, columns, col_names, obj):
        """Configura e resolve o problema mestre como LP."""
        master = Cplex()
        master.set_problem_type(master.problem_type.LP)

        # Definição da função objetivo
        master.objective.set_sense(master.objective.sense.maximize)
        master.variables.add(obj=obj, names=col_names, lb=[0.0] * len(columns), ub=[1.0] * len(columns))

        # Restrições: cada vértice deve estar coberto exatamente uma vez
        constraints = []
        rhs = []
        senses = []

        for v in self.vertices:
            cols_covering_v = []
            coeffs = []
            for idx, col in enumerate(columns):
                if col[v] == 1:
                    cols_covering_v.append(idx)
                    coeffs.append(1.0)
            constraints.append(SparsePair(ind=cols_covering_v, val=coeffs))
            rhs.append(1.0)
            senses.append("E")

        master.linear_constraints.add(lin_expr=constraints, senses=senses, rhs=rhs)

        # Resolver o problema mestre
        try:       
            master.parameters.mip.cuts.mircut.set(-1)
            master.parameters.mip.cuts.implied.set(-1)
            master.parameters.mip.cuts.gomory.set(-1)
            master.parameters.threads.set(1)
            master.parameters.clocktype.set(1)
            master.parameters.preprocessing.presolve.set(0)
            master.parameters.preprocessing.boundstrength.set(0)
            master.parameters.preprocessing.coeffreduce.set(0)
            master.parameters.preprocessing.relax.set(0)
            master.parameters.preprocessing.aggregator.set(0)
            master.parameters.preprocessing.reduce.set(0)
            master.parameters.preprocessing.reformulations.set(0)
            master.solve()
            objective_value = master.solution.get_objective_value()
            print(f"Função objetivo (mestre): {objective_value}")
            self.dual_values = master.solution.get_dual_values()

            print("Valores duais:", self.dual_values)
        except CplexError as e:
            print("Erro ao resolver o problema mestre:", e)
            return None

        return columns, objective_value  # Retorna as variáveis e a função objetivo



    def solve_subproblem(self, color):
        """Configura e resolve o subproblema para precificação."""
        sub = Cplex()
        sub.objective.set_sense(sub.objective.sense.maximize)

        # Variáveis y_i e h_i
        y_vars = [f"y_{v}" for v in self.vertices]
        h_vars = [f"h_{v}" for v in self.vertices]

        # Objetivo: maximize sum(h_i) - sum(mu_i * y_i)
        mu = self.dual_values  # Lista de mu_i
        obj_y = [-mu[v] for v in self.vertices]
        obj_h = [1.0] * len(self.vertices)
        sub.variables.add(
            obj=obj_y + obj_h,
            names=y_vars + h_vars,
            types=["B"] * (2 * len(self.vertices)),
            lb=[0.0] * (2 * len(self.vertices)),
            ub=[1.0] * (2 * len(self.vertices))
        )

        # Restrições para pré-coloridos
        for v in self.vertices:
            if v in self.precolored:
                if self.precolored[v] == color:
                    sub.linear_constraints.add(
                        lin_expr=[SparsePair(ind=[y_vars[v]], val=[1.0])],
                        senses=["E"],
                        rhs=[1.0]
                    )
                else:
                    sub.linear_constraints.add(
                        lin_expr=[SparsePair(ind=[y_vars[v]], val=[1.0])],
                        senses=["E"],
                        rhs=[0.0]
                    )

        # Restrições: h_i <= y_i
        for v in self.vertices:
            sub.linear_constraints.add(
                lin_expr=[SparsePair(ind=[h_vars[v], y_vars[v]], val=[1.0, -1.0])],
                senses=["L"],
                rhs=[0.0]
            )

        # Restrições: h_i <= y_j para todos os vizinhos j de i
        for v in self.vertices:
            for j in self.adj_list[v]:
                sub.linear_constraints.add(
                    lin_expr=[SparsePair(ind=[h_vars[v], y_vars[j]], val=[1.0, -1.0])],
                    senses=["L"],
                    rhs=[0.0]
                )

        # Restrições: h_i >= sum(y_j para todos os vizinhos j de i) - |Γ(i)| + y_i
        for v in self.vertices:
            neighbors = self.adj_list[v]
            if neighbors:
                ind = [h_vars[v]] + [y_vars[j] for j in neighbors] + [y_vars[v]]
                val = [1.0] + [-1.0] * len(neighbors) + [1.0]
                sub.linear_constraints.add(
                    lin_expr=[SparsePair(ind=ind, val=val)],
                    senses=["G"],
                    rhs=[-len(neighbors)]
                )
            else:
                # Caso o vértice não tenha vizinhos
                sub.linear_constraints.add(
                    lin_expr=[SparsePair(ind=[h_vars[v], y_vars[v]], val=[1.0, -1.0])],
                    senses=["G"],
                    rhs=[0.0]
                )

        # Resolver o subproblema
        try:
            sub.solve()
            obj_val = sub.solution.get_objective_value()
            print(f"Função objetivo (subproblema): {obj_val}")
            if obj_val > 1e-5:
                y_values = sub.solution.get_values(y_vars)
                h_values = sub.solution.get_values(h_vars)
                r = [v for v in self.vertices if y_values[v] > 0.5]
                hkr = int(round(sum(h_values)))
                print(f"Nova coluna para cor {color}: r={r}, h_k(r)={hkr}")
                return (f"lambda_new_{color}", r, hkr)
            else:
                return None
        except CplexError as e:
            print("Erro ao resolver o subproblema:", e)
            return None
    def column_generation(self):
        """Executa a geração de colunas."""
        columns, col_names, obj = self.initialize_columns()
        master = self.solve_master_problem(columns, col_names, obj)
        if master is None:
            return

        while True:
            new_columns = []
            for color in range(1, self.k + 1):
                new_col = self.solve_subproblem(color)
                if new_col:
                    new_columns.append(new_col)

            if not new_columns:
                print("Nenhuma nova coluna encontrada.")
                print(f"Valor final: {master_obj_value}")
                break

            for name, r, hkr in new_columns:
                col = [1 if v in r else 0 for v in self.vertices]
                columns.append(col)
                col_names.append(name)
                obj.append(hkr)

            columns, master_obj_value = self.solve_master_problem(columns, col_names, obj)

        print("Colunas finais:", col_names)
        return columns, master_obj_value


########################################################### Formulaçã Original - Resolvendo com B&B pelo solver - Baseline de comparação #####################################################:

class MaxHVOriginalFormulation:
    def __init__(self, adj_list, precolored, vertices, k):
        self.adj_list = adj_list
        self.precolored = precolored
        self.vertices = vertices
        self.k = k

    def solve(self):
        """Resolve o problema MaxHV usando a formulação original."""
        try:
            model = Cplex()
            model.set_problem_type(model.problem_type.LP)
            model.objective.set_sense(model.objective.sense.maximize)

            # Adicionar variáveis x_{ik} e h~i
            x_vars = {}
            h_vars = {}

            for i in self.vertices:
                h_name = f"h_{i}"
                h_vars[i] = h_name
                model.variables.add(
                    obj=[-1],  # Coeficiente negativo porque queremos minimizar o número de vértices infelizes
                    lb=[0], 
                    ub=[1], 
                    types="B",
                    names=[h_name]
                )

                for k in range(1, self.k + 1):
                    x_name = f"x_{i}_{k}"
                    x_vars[(i, k)] = x_name
                    model.variables.add(
                        obj=[0], 
                        lb=[0], 
                        ub=[1], 
                        types="B",
                        names=[x_name]
                    )

            # Restrições 1: Pré-coloração
            for i, color in self.precolored.items():
                x_name = x_vars[(i, color)]
                model.linear_constraints.add(
                    lin_expr=[([x_name], [1])],  # A forma correta de passar as variáveis e coeficientes
                    senses="E",
                    rhs=[1]
                )

            # Restrições 2: Cada vértice deve receber exatamente uma cor
            for i in self.vertices:
                x_names = [x_vars[(i, k)] for k in range(1, self.k + 1)]
                model.linear_constraints.add(
                    lin_expr=[(x_names, [1] * len(x_names))],  # A forma correta de passar as variáveis e coeficientes
                    senses="E",
                    rhs=[1]
                )

            for i in self.vertices:
                for j in self.adj_list[i]:
                    for k in range(1, self.k + 1):
                        x_i_k = x_vars[(i, k)]
                        x_j_k = x_vars[(j, k)]
                        h_i = h_vars[i]

                        # h_i >= x_i_k - x_j_k
                        model.linear_constraints.add(
                            lin_expr=[([h_i, x_i_k, x_j_k], [1, -1, 1])],  # A forma correta de passar as variáveis e coeficientes
                            senses="G",
                            rhs=[0]
                        )

                        # h_i >= x_j_k - x_i_k
                        model.linear_constraints.add(
                            lin_expr=[([h_i, x_i_k, x_j_k], [1, 1, -1])],  # A forma correta de passar as variáveis e coeficientes
                            senses="G",
                            rhs=[0]
                        )

            # Resolver o modelo

            model.parameters.mip.strategy.heuristicfreq.set(-1)
            model.parameters.mip.cuts.mircut.set(-1)
            model.parameters.mip.cuts.implied.set(-1)
            model.parameters.mip.cuts.gomory.set(-1)
            model.parameters.mip.cuts.flowcovers.set(-1)
            model.parameters.mip.cuts.pathcut.set(-1)
            model.parameters.mip.cuts.liftproj.set(-1)
            model.parameters.mip.cuts.zerohalfcut.set(-1)
            model.parameters.mip.cuts.cliques.set(-1)
            model.parameters.mip.cuts.covers.set(-1)
            model.parameters.threads.set(1)
            model.parameters.clocktype.set(1)
            model.parameters.timelimit.set(60)
            model.parameters.preprocessing.presolve.set(0)
            model.parameters.preprocessing.boundstrength.set(0)
            model.parameters.preprocessing.coeffreduce.set(0)
            model.parameters.preprocessing.relax.set(0)
            model.parameters.preprocessing.aggregator.set(0)
            model.parameters.preprocessing.reduce.set(0)
            model.parameters.preprocessing.reformulations.set(0)
            model.solve()

            # Exibir resultados
            objective_value = model.solution.get_objective_value()
            print(f"Função objetivo (número de vértices felizes): {len(self.vertices) + objective_value}")
            solution_values = {}
            for i in self.vertices:
                is_unhappy = model.solution.get_values(h_vars[i])
                print(f"Vértice {i}: {'Infeliz' if is_unhappy > 0.5 else 'Feliz'}")
                solution_values[i] = {
                    'h_i': is_unhappy,
                    'assigned_colors': []
                }
                for k in range(1, self.k + 1):
                    if model.solution.get_values(x_vars[(i, k)]) > 0.5:
                        print(f"  Cor atribuída: {k}")
                        solution_values[i]['assigned_colors'].append(k)

            objective_value_final = len(self.vertices) + objective_value

            return solution_values, objective_value_final  # Retorna as variáveis e a função objetivo

        except CplexError as e:
            print("Erro ao resolver o modelo:", e)
            return None, None


    def plot_objective_values(self):
        # Example plot for objective values
        # Assuming self.objective_values is a list or array of objective values
        plt.plot(self.objective_values)
        plt.title('Objective Values')
        plt.xlabel('Iteration')
        plt.ylabel('Objective Value')
        plt.show()


# Exemplo de Uso
if __name__ == "__main__":
    # Executar a geração de colunas
     
    # TROCAR O PATH DA INSTANCIA!!!
    
    arquivo = '/home/rafael/Documents/HappySet/MIHS/inputs/testes/gercol_instance.txt'
    instancia = GraphInstance(arquivo)

    adj_list, precolored, vertices, arestas, k = instancia.get_data()

    print("Dados da instância")
    print("Lista de Adjacências:")
    print(adj_list)

    print("\nVertices pré-coloridos:")
    print(precolored)

    print("\nVertices:")
    print(vertices)

    print("\nArestas:")
    print(arestas)

    print("\nValor de k:")
    print(k)
    print()
    print("###### INICIANDO GERAÇÃO DE COLUNAS ######")
    print()
    maxhv = MaxHVColumnGeneration(adj_list, precolored, vertices, k)
    maxhv.column_generation()
    print()
    print("###################################################### INICIANDO Formulação original ######################################################")
    print()
    maxhv_orig = MaxHVOriginalFormulation(adj_list, precolored, vertices, k)
    maxhv_orig.solve()

    # compare_solutions_and_plot(adj_list, precolored, vertices, k)