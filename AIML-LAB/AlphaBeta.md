# Minimax Algorithm with Alpha-Beta Pruning

## Code
```python
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
    if depth == 0 or not flat_tree:  # Base case: depth 0 or flat_tree empty
        return flat_tree.pop(0) if flat_tree else None  # Return None if list is empty
    children = []
    for _ in range(2):  # Build 2 children for binary tree
        if flat_tree:  # Check if flat_tree has nodes left
            children.append(build_tree(flat_tree, depth - 1))
        else:
            children.append(None)  # Add None for missing nodes
    return children


flattened_tree = list(map(int, input("Enter the flattened game tree (space-separated): ").split()))
depth = int(input("Enter the depth of the tree: "))
game_tree = build_tree(flattened_tree, depth)
evaluated_value = minimax(game_tree, depth, float('-inf'), float('inf'), True)
print("Evaluated Value:", evaluated_value)
```

## Output Examples
```
Enter the flattened game tree (space-separated): 6 7 2 4 1 8 10 9 25 4 2 5 7 8 12 5
Enter the depth of the tree: 4
Evaluated Value: 7
```

## Explanation of the Program

### 1. **Minimax Algorithm with Alpha-Beta Pruning**
The function `minimax(node, depth, alpha, beta, maximizingPlayer)` recursively evaluates the best possible move in a given game tree.

- **Base Case:** If the depth reaches 0 or the node is a numerical value (terminal state), return the node value.
- **Maximizing Player:**
  - Initialize `maxEva` to negative infinity.
  - Recursively compute the best possible move.
  - Update `alpha` and prune unnecessary nodes when `beta <= alpha`.
- **Minimizing Player:**
  - Initialize `minEva` to positive infinity.
  - Recursively compute the worst possible move for the maximizing player.
  - Update `beta` and prune unnecessary nodes when `beta <= alpha`.

### 2. **Building the Game Tree**
The function `build_tree(flat_tree, depth)` constructs a binary game tree from a **flat list of integers**.

- If `depth == 0` or the list is empty, return the next available value.
- Recursively construct children nodes for a binary tree.
- If no values are left, assign `None`.

### 3. **User Input Handling**
- The user provides a **space-separated list** representing a flattened game tree.
- The user specifies the **depth of the tree**.

### 4. **Evaluating the Tree**
- The game tree is built using `build_tree(flattened_tree, depth)`.
- The `minimax` function is called to evaluate the best possible outcome for the maximizing player.

## Advantages and Disadvantages of Minimax with Alpha-Beta Pruning

### **Advantages:**
1. **Efficient Pruning:** Reduces the number of nodes evaluated, improving performance.
2. **Optimal Play:** Ensures the best possible decision is made.
3. **Works for Various Games:** Can be applied to Chess, Tic-Tac-Toe, Connect Four, etc.

### **Disadvantages:**
1. **Computational Cost:** Still expensive for large game trees.
2. **Requires Full Game Tree:** Works best with a fully known and deterministic environment.
3. **Limited by Depth:** Deep trees may still be computationally expensive.

## Applications of Minimax with Alpha-Beta Pruning
1. **Chess AI:** Used in decision-making to prune unnecessary moves.
2. **Tic-Tac-Toe AI:** Evaluates best moves to win or force a draw.
3. **Strategy-Based Games:** Applied in turn-based strategy games like Connect Four.
4. **Automated Decision-Making:** Used in game simulations and AI competitions.

## Few Questions and Answers

### **1. What is the purpose of Alpha-Beta pruning in Minimax?**
Alpha-Beta pruning eliminates branches in the game tree that do not need to be evaluated, making the algorithm faster without changing the final result.

### **2.What is the worst-case and best-case time complexity of Minimax with Alpha-Beta Pruning?
Best case: O(𝑏^(d/2)), Worst case: O(𝑏^𝑑), where b = branching factor, d = depth

### **3.Why does Alpha-Beta Pruning perform best in depth-first search (DFS) rather than breadth-first search (BFS)?
DFS evaluates a path completely before moving to the next, leading to early pruning
