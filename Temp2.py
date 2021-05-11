import os
import networkx as nx
import pygraphviz
import pandas as pd
import matplotlib.pyplot as plt


def dot_to_graph(dir_path):
    f = []

    # get dot file lists with path
    for (path, directory, files) in os.walk(dir_path):
        for filename in files:
            ext = os.path.splitext(filename)[-1]
            if ext == '.dot':
                tmp = path + "\\" + filename
                f.append(tmp)

    graphs = []
    graphs2 = []

    # convert dot file to nxgraph
    for file in f:
        g = nx.drawing.nx_agraph.to_agraph(nx.drawing.nx_pydot.read_dot(file))  # dot -> Agraph
        g = nx.to_networkx_graph(g)  # Agraph -> networkX graph
        g = nx.convert_node_labels_to_integers(g)  # change node name str to int for processing dataset
        for i in range(g.number_of_nodes()):
            g.add_edge(i, i)  # add self loop
        graphs.append(g)

        # remove duplicate edges and if duplicate edges then add weight
        tmp = nx.Graph()
        for u,v,data in g.edges(data=True):
            w = data['weight'] if 'weight' in data else 1.0
            if tmp.has_edge(u,v):
                tmp[u][v]['weight'] += w
            else:
                tmp.add_edge(u, v, weight=w)
        # print(tmp.edges(data=True))
        nx.draw(tmp, with_labels=True, font_weigth="bold")
        plt.show()
        exit()
        graphs2.append(tmp)

    # graph_to_csv(graphs2)


def graph_to_csv(graphs):
    i = 0  # for count graphid

    # graph to pandas dataframe + add graphid column in dataframe
    for graph in graphs:
        df = nx.to_pandas_edgelist(graph)  # for graph edges dataset
        df.insert(0, "graph_id", i, True)  # add graph_id at first column in dataframe
        # df.insert(3, "feat", 0, True)

        # labeling each group in properties
        if i < 43:
            k = 1  # Backdoor
        elif i < 45:
            k = 2  # Botnets
        elif i < 52:
            k = 3  # Infectors
        elif i < 395:
            k = 4  # Mirai-Family
        elif i < 429:
            k = 5  # Rootkits
        elif i < 493:
            k = 6  # Tools
        else:
            k = 7  # Trojans

        # dataframe for graph properties datatset
        df2 = pd.DataFrame({"graph_id": [i], "label": [k], "num_nodes": [len(graph)]})

        # if it is first graph, then create new csv file with column headers
        if i == 0:
            df.to_csv(".\\CallGraphEdges3.csv", mode="w", index=False, header=True)
            df2.to_csv(".\\CallGraphProperties3.csv", mode="w", index=False, header=True)
        else:  # if is not, then append graph to csv file with data only
            df.to_csv(".\\CallGraphEdges3.csv", mode="a", index=False, header=False)
            df2.to_csv(".\\CallGraphProperties3.csv", mode="a", index=False, header=False)

        i += 1

if __name__ == "__main__":
    d = r"C:\Users\jw\Graphs\CG_dot"
    dot_to_graph(d)
