import networkx as nx

# 创建有向图
G = nx.DiGraph()
edges = [('A', 'B'), ('B', 'C'), ('A', 'C'), ('A', 'D'), ('D', 'E'), ('X', 'Y')]
G.add_edges_from(edges)

# 计算度中心性
degree_centrality = nx.degree_centrality(G)

# 计算接近度中心性
closeness_centrality = nx.closeness_centrality(G)

# 计算中介中心性
betweenness_centrality = nx.betweenness_centrality(G)

# 显示结果
print("Degree Centrality:", degree_centrality)
print("Closeness Centrality:", closeness_centrality)
print("Betweenness Centrality:", betweenness_centrality)
