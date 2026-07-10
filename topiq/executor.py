def execute_graph(graph, inputs):
    ctx = inputs.copy()
    for node in graph:
        op = node["op"]
        out = node["output"]
        args = node.get("args", [])

        vals = []
        for a in args:
            if isinstance(a, str) and a in ctx:
                vals.append(ctx[a])
            else:
                vals.append(a)

        obj = vals[0]

        if op == "add":
            res = obj + vals[1]
        elif op == "sub":
            res = obj - vals[1]
        elif op == "mul":
            res = obj * vals[1]
        elif op == "div":
            res = obj / vals[1]
        elif op == "pow":
            res = obj ** vals[1]
        elif op == "sum":
            res = obj.sum(*vals[1:])
        elif op == "mean":
            res = obj.mean(*vals[1:])
        elif op == "sigmoid":
            res = obj.sigmoid()
        elif op == "relu":
            res = obj.relu()
        elif op == "square":
            res = obj ** 2
        else:
            raise ValueError(f"Unknown op: {op}")

        ctx[out] = res
    return ctx
