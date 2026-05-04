from .inline_comp_graph import graph
try:
    from .inline_training_graphs import plot, scatter
except ImportError:
    pass
