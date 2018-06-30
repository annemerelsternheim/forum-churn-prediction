import sys
from tqdm import *
import copy

def dijkstra(G,start,goal):
    shortest_distance = {}
    predecessor = {}
    unseenNodes = copy.deepcopy(G)
    infinity = 999999
    path = []
    for node in unseenNodes:
        shortest_distance[node] = infinity
    shortest_distance[start]=0
    
    while unseenNodes:
        minNode = None
        for node in unseenNodes:
            if minNode is None:
                minNode = node
            elif shortest_distance[node] < shortest_distance[minNode]:
                minNode = node
        
        # main part of algorithm
        for childNode, weight in graph[minNode].items():
            if weight + shortest_distance[minNode] < shortest_distance[childNode]:
                shortest_distance[childNode] = weight + shortest_distance[minNode]
                predecessor[childNode] = minNode
        unseenNodes.pop(minNode)
    
    currentNode = goal
    while currentNode != start:
        try:
            path.insert(0,currentNode)
            currentNode = predecessor[currentNode]
        except KeyError:
            #print 'path not reachable'
            return None
            break
    path.insert(0,start)
    if shortest_distance[goal] != infinity:
        #print 'shortest distance is ' + str(shortest_distance[goal])
        #print 'and the path is ' + str(path)
        return path

#---------------------------------------------------------
# CODE
#---------------------------------------------------------

if sys.argv[1] == 'graph':
    graph = {'a':{'b':10,'c':3},'b':{'c':1,'d':2},'c':{'b':4,'d':8,'e': 2},'d':{'e':7},'e':{'d':9}}
    print 'using default graph'
else:
    graph = sys.argv[1]

closeness_centrality = dict()
graph = json.load(open('graph.json'))

with tqdm(total=len(graph)) as pbar:
    for user_i in graph: # 893 users
        close = []
        pbar.update(1)
        for user_j in graph:
            if user_i != user_j:
                path = dijkstra(graph,user_i,user_j)
                length = float('nan') if path is None else len(path) 
                close.append(length)
        print close
        closeness_centrality[user_i] = np.nanmean(close)
print closeness_centrality
