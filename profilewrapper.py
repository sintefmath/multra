def profilethis(fun,figurename='basic.png'):
    "usage example: >> from linear_1d include * >> profilewrapper(lambda: linear(1000,1.,2))"
    from pycallgraph import PyCallGraph
    from pycallgraph.output import GraphvizOutput
    graphviz = GraphvizOutput()
    graphviz.output_file = figurename
    with PyCallGraph(output=graphviz):
        print("Starting profiling...")
        fun()
        print("... done.")
