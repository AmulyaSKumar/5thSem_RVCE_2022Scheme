def minimax(node, depth, alpha, beta, maximizingPlayer):
    if depth == 0 or isinstance(node, int): return node
    if maximizingPlayer:
        maxEva = float('-inf')
        for child in node:
            maxEva = max(maxEva, minimax(child, depth - 1, alpha, beta, False))
            alpha = max(alpha, maxEva)
            if beta <= alpha: break
        return maxEva
    else:
        minEva = float('inf')
        for child in node:
            minEva = min(minEva, minimax(child, depth - 1, alpha, beta, True))
            beta = min(beta, minEva)
            if beta <= alpha: break
        return minEva

def build_tree(flat_tree, depth):
    if depth == 0:
        return flat_tree.pop(0)
    children = []
    for _ in range(2):
        children.append(build_tree(flat_tree, depth - 1))
    return children

flattened_tree = list(map(int, input("Enter the flattened game tree (space-separated): ").split()))
depth = int(input("Enter the depth of the tree: "))
game_tree = build_tree(flattened_tree, depth)
evaluated_value = minimax(game_tree, depth, float('-inf'), float('inf'), True)
print("Evaluated Value:", evaluated_value)
