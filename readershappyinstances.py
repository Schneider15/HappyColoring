import os
import re
import openpyxl
import networkx as nx
import matplotlib.pyplot as plt

def get_files_in_folder(folder_path):
    """Obtém todos os arquivos da pasta especificada."""
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    return files

def save_to_excel(output_file, data, headers=None):
    """Salva os dados processados em um arquivo Excel."""
    directory = os.path.dirname(output_file)
    
    # Criação do diretório se não existir
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Verificação se o arquivo já existe
    if not os.path.exists(output_file):
        try:
            # Criação do arquivo Excel
            workbook = openpyxl.Workbook()
            sheet = workbook.active
            if headers:
                sheet.append(headers)  # Adiciona o cabeçalho
            workbook.save(output_file)
            print(f'Arquivo Excel criado com sucesso: {output_file}')
        except Exception as e:
            print(f'Erro ao criar o arquivo Excel: {e}')
            return

    try:
        # Carregar ou reabrir o arquivo Excel existente
        workbook = openpyxl.load_workbook(output_file)
        sheet = workbook.active
        
        # Adiciona os dados ao final da planilha
        for row in data:
            sheet.append(row)
        
        workbook.save(output_file)
        print(f'Dados adicionados ao arquivo Excel {output_file}')
    except Exception as e:
        print(f'Erro ao manipular o arquivo Excel: {e}')

def draw_graph(G, color_assignments, file_name=None):
    """Desenha o grafo com as cores atribuídas aos vértices."""
    pos = nx.spring_layout(G)  # Layout para a posição dos vértices
    colors = [color_assignments.get(node) for node in G.nodes()]
    nx.draw(G, pos, with_labels=True, node_color=colors, cmap=plt.get_cmap('tab10'))
    
    # Define o caminho para a pasta de Downloads
    downloads_path = os.path.expanduser("~/Downloads")
    output_file = os.path.join(downloads_path, f"{file_name}.png")
    
    if file_name:
        # Salva a imagem do grafo com o nome especificado
        plt.savefig(output_file)
        plt.close()  # Fecha o gráfico
        print(f'Imagem do grafo salva em: {output_file}')

class GraphInstance:
    def __init__(self, filename):
        self.filename = filename
        self.adj_list = None
        self.precolored = None
        self.num_vertices = None
        self.arestas_list = None
        self.k = None
        self.seed = None
        self.density = None
        self.degrees = None
        self.vertices = None  # A variável que armazenará o conjunto de vértices
        self._read_instance()

    def _read_instance(self):
        """Método privado para ler e processar a instância do arquivo."""
        try:
            with open(self.filename, 'r') as file:
                linhas = file.readlines()

            # Processar as estatísticas do grafo
            self._parse_graph_stats(linhas)

            # Processar as arestas e lista de adjacências
            self._parse_edges(linhas)

            # Processar os vértices pré-coloridos
            self._parse_precolored(linhas)

            # Armazenar o conjunto de vértices (todos os vértices)
            self.vertices = list(range(self.num_vertices))  # Todos os vértices são armazenados aqui

        except Exception as e:
            print(f'Erro ao ler o arquivo {self.filename}: {e}')

    def _parse_graph_stats(self, linhas):
        """Extrair informações gerais do grafo como número de vértices, k, e estatísticas adicionais."""
        for line in linhas:
            if line.startswith('c Number of Nodes'):
                self.num_vertices = int(line.split('=')[1].strip())
            elif line.startswith('c Number of colours k'):
                self.k = int(line.split('=')[1].strip())
            elif line.startswith('c Seed'):
                self.seed = int(line.split('=')[1].strip())
            elif line.startswith('c Actual density'):
                self.density = float(line.split('=')[1].strip())
            elif line.startswith('c The degrees of nodes'):
                self.degrees = list(map(int, re.findall(r'\d+', line)))

    def _parse_edges(self, linhas):
        """Ler as arestas e montar a lista de adjacências."""
        self.adj_list = [[] for _ in range(self.num_vertices)]
        self.arestas_list = []

        start_edge = False
        for line in linhas:
            if line.startswith('p edge'):
                start_edge = True
            elif start_edge and line.startswith('e'):
                v1, v2 = map(int, line.split()[1:])
                self.adj_list[v1 - 1].append(v2 - 1)
                self.adj_list[v2 - 1].append(v1 - 1)
                self.arestas_list.append((v1 - 1, v2 - 1))

    def _parse_precolored(self, linhas):
        """Extrair a lista de vértices pré-coloridos."""
        self.precolored = {}
        for line in linhas:
            if line.startswith('n'):
                v, c = map(int, line.split()[1:])
                self.precolored[v - 1] = c

    def get_data(self):
        """Retorna todos os dados processados."""
        return self.adj_list, self.precolored, self.vertices, self.arestas_list, self.k

    def get_instance_info(self):
        """Retorna um resumo das informações do grafo."""
        adj_list_str = ', '.join([str(neighbors) for neighbors in self.adj_list])
        precolored_str = ', '.join([f"{v}:{c}" for v, c in self.precolored.items()])
        vertices_str = ', '.join(map(str, self.vertices))  # Agora retorna os vértices
        arestas_str = ', '.join([f"{a}-{b}" for a, b in self.arestas_list])
        return [self.filename, len(self.vertices), len(self.arestas_list), adj_list_str, precolored_str, vertices_str, arestas_str, self.k]

    def get_colored_vertices(self):
        """Retorna o conjunto de vértices pré-coloridos cuja cor é menor ou igual a k."""
        if self.k is None:
            raise ValueError("Valor de k não foi definido.")
        
        # Retorna os vértices cujas cores são <= k
        return [v for v, color in self.precolored.items() if color <= self.k]

# Função para processar todas as instâncias de uma pasta
def process_instances_from_folder(folder_path):
    instances = sorted(get_files_in_folder(folder_path))
    print(f'Instâncias encontradas: {instances}')
    
    for instance_file in instances:
        arquivo_path = os.path.join(folder_path, instance_file)
        print(f'Processando instância: {arquivo_path}')
        
        # Criar instância do grafo
        instancia = GraphInstance(arquivo_path)
        adj_list, precolored, vertices, arestas, k = instancia.get_data()

        # Exibir informações
        print(f"Instância: {arquivo_path}")
        print("Lista de Adjacências:")
        print(adj_list)
        print("\nVértices pré-coloridos:")
        print(precolored)
        print("\nVértices (todos os vértices):")
        print(vertices)  # Exibindo os vértices
        print("\nArestas:")
        print(arestas)
        print("\nValor de k:")
        print(k)

        # Obter vértices pré-coloridos cujas cores são <= k
        colored_vertices = instancia.get_colored_vertices()
        print(f"\nVértices pré-coloridos com cor <= {k}: {colored_vertices}")

        # Opcionalmente, salvar em Excel
        # output_file = '/home/rafael/Documents/HappySet/MIHS/outputs/graph_data.xlsx'
        # headers = ["Arquivo", "Número de Vértices", "Número de Arestas", "Lista de Adjacências", "Vértices Pré-coloridos", "Vértices", "Arestas", "k"]
        # data = [instancia.get_instance_info()]
        # save_to_excel(output_file, data, headers)

# Função principal
def main():
    # Caminho da pasta contendo as instâncias
    folder_path = '/home/rafael/Documents/HappySet/MIHS/inputs/happygen/output/testes'

    # Processar todas as instâncias na pasta
    process_instances_from_folder(folder_path)

# Execução do código principal apenas se o script for rodado diretamente
if __name__ == "__main__":
    main()
