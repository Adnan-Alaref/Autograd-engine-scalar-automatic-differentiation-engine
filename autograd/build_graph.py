from graphviz import Digraph

'''graphviz in Python is a powerful library used to create, visualize,and render graphs
  (nodes and edges) — such as computational graphs, decision trees, and neural network diagrams.'''

def trace(root):

  # builds a set of all nodes and edges (set(tuple)) in a graph
  nodes, edges = set(), set()
  
  def build_graph(n):
    if n not in nodes:
      nodes.add(n)
      for _child in n._prev:
        edges.add((_child,n))
        build_graph(_child)
  build_graph(root)
  return nodes, edges

def draw_graph(root):
  dot = Digraph(name="AutoGrad",
                format='svg',
                comment="Computational Graph",
                graph_attr={'rankdir':'LR', 'bgcolor': 'white'},
                edge_attr={'penwidth': '1', 'color': 'black', 'arrowhead': 'vee'})

  nodes, edges = trace(root)
  for n in nodes:
    uid = str(id(n)) # Unique memmory address(Identifier)
    # for any value in the graph, create a rectangular ('record') node for it
    dot.node(name=uid, label="{%s | data %.4f | grad %.4f}"%(n.label, n.data, n.grad), shape='record', fillcolor='lightgray')
    # dot.node(name=uid, label=f"{n.label} | {n.data:.4f} | {n.grad:.4f}", shape='Mrecord',fillcolor='lightgray')

    if n._op:
       # if this value is a result of some operation, create an op node for it
      dot.node(name = uid + n._op ,label = n._op)
       # and connect this node to it
      dot.edge(tail_name=uid+n._op, head_name=uid)

  for (e1, e2) in edges:
    # connect n1 to the op node of n2
    head = str(id(e2)) + (e2._op if e2._op else '')
    dot.edge(tail_name=str(id(e1)), head_name=head)
  
  # Generate multiple output formats
  dot.render('graph_output', format='png', cleanup=True)
  return dot