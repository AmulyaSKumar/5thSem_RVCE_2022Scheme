# A* Algorithm for Solving 8-Puzzle Problem

## About the A* Algorithm
A* is a graph traversal and search algorithm that finds the shortest path from a start node to a target node. It uses two key components:
- **g(n)**: The cost to reach the current node.
- **h(n)**: The estimated cost to reach the goal from the current node.

The total cost function is \( f(n) = g(n) + h(n) \). The algorithm explores paths with the lowest \( f(n) \) first, ensuring both efficiency and optimality when the heuristic is admissible.


## Code
```python
import heapq

def misplaced_tiles(state, goal):
    count = 0
    for i in range(3):
        for j in range(3):
            if state[i][j] != 0 and state[i][j] != goal[i][j]:
                count += 1
    return count

def get_neighbors(state):
    neighbors = []
    x, y = next((i, j) for i in range(3) for j in range(3) if state[i][j] == 0)
    moves = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]
    for nx, ny in moves:
        if 0 <= nx < 3 and 0 <= ny < 3:
            new_state = [list(row) for row in state]
            new_state[x][y], new_state[nx][ny] = new_state[nx][ny], new_state[x][y]
            neighbors.append(tuple(tuple(row) for row in new_state))
    return neighbors

def a_star(start, goal):
    open_set = []
    heapq.heappush(open_set, (0, start, 0, None))
    visited = set()
    parents = {}

    while open_set:
        _, current, g, parent = heapq.heappop(open_set)
        if current in visited:
            continue
        visited.add(current)
        parents[current] = parent

        if current == goal:
            path = []
            while current:
                path.append(current)
                current = parents[current]
            return path[::-1]

        for neighbor in get_neighbors(current):
            if neighbor not in visited:
                h = misplaced_tiles(neighbor, goal)
                heapq.heappush(open_set, (g + 1 + h, neighbor, g + 1, current))

    return None

def parse_input(input_string):
    try:
        numbers = list(map(int, input_string.split()))
        if sorted(numbers) != list(range(9)):
            raise ValueError("Input must contain numbers 0 through 8 exactly once.")
        return tuple(tuple(numbers[i:i + 3]) for i in range(0, 9, 3))
    except Exception as e:
        print("Invalid input:", e)
        return None

# Main Program
if __name__ == "__main__":
    print("Enter the initial state of the 8-puzzle (e.g., '1 2 3 4 0 5 6 7 8'):")
    initial_input = input().strip()
    print("Enter the goal state of the 8-puzzle (e.g., '1 2 3 4 5 6 7 8 0'):")
    goal_input = input().strip()

    start_state = parse_input(initial_input)
    goal_state = parse_input(goal_input)

    if start_state and goal_state:
        solution = a_star(start_state, goal_state)
        if solution:
            print("Solution found!")
            for step in solution:
                for row in step:
                    print(row)
                print()
        else:
            print("No solution exists.")
    else:
        print("Failed to process input. Please try again.")
```

---

## Example  Output

```
Enter the initial state of the 8-puzzle (e.g., '1 2 3 4 0 5 6 7 8'):
1 2 3 4 0 5 6 7 8
Enter the goal state of the 8-puzzle (e.g., '1 2 3 4 5 6 7 8 0'):
1 2 3 4 5 6 7 8 0

Solution found!
(1, 2, 3)
(4, 0, 5)
(6, 7, 8)

(1, 2, 3)
(4, 5, 0)
(6, 7, 8)

(1, 2, 3)
(4, 5, 6)
(7, 8, 0)
```

---

## Advantages and Disadvantages of A* Algorithm

### Advantages:
1. **Optimality**: A* guarantees an optimal solution if the heuristic is admissible and consistent.
2. **Efficiency**: It uses a heuristic to focus the search, reducing the search space.
3. **Flexibility**: A* can be adapted to solve a wide variety of problems by changing the heuristic function.

### Disadvantages:
1. **Memory Intensive**: A* requires significant memory to store all visited nodes and their associated costs.
2. **Time Complexity**: For large search spaces, A* can become slow if the heuristic is not well-designed.
3. **Dependence on Heuristic**: The performance heavily relies on the quality of the heuristic function.

---

## Few Questions
1. **What is the A * algorithm, and why is it widely used?**
   - A* is a pathfinding and graph traversal algorithm that guarantees an optimal solution by combining cost-so-far (g) and heuristic cost (h). It is widely used due to its efficiency and versatility.

2. **What is the role of the heuristic function in the A * algorithm?**
   - The heuristic function estimates the cost from the current state to the goal, guiding the algorithm to explore promising paths.

3. **Explain the difference between admissible and consistent heuristics.**
   - Admissible heuristics never overestimate the actual cost, ensuring optimality. Consistent heuristics satisfy the triangle inequality and guarantee efficient search without revisiting nodes.

4. **What is the time and space complexity of the A * algorithm?**
   - Time complexity: ( O(b^d) ) in the worst case, where ( b) is the branching factor and ( d) is the depth of the solution.
   - Space complexity:( O(b^d) ) as it stores all explored and frontier nodes.

5. **How does the `misplaced_tiles` heuristic work, and why is it effective for the 8-puzzle problem?**
   - It counts the number of tiles out of place compared to the goal state. It is simple, fast to compute, and provides a reasonable estimate of the remaining cost.

6. **Why does the A * algorithm guarantee an optimal solution?**
   - A* guarantees optimality if the heuristic is admissible, ensuring it never overestimates the true cost to reach the goal.

7. **What happens if the heuristic function is not admissible?**
   - If the heuristic overestimates the cost, A* may find a suboptimal solution, as it might prune valid paths prematurely.

8. **How does A* differ from other search algorithms like BFS or Dijkstra's?**
   - BFS explores all paths equally, Dijkstra's focuses on the shortest path based on cost-so-far, while A* combines cost-so-far and estimated cost, making it more efficient.

9. **Can A* handle negative edge weights? Why or why not?**
    - No, A* cannot handle negative edge weights as it assumes all edge costs are non-negative. Negative weights may lead to incorrect estimates and suboptimal solutions.

10. How does iterative deepening A (IDA) improve upon traditional A*?**
    - IDA* reduces memory usage by performing depth-limited searches iteratively.

- **Applications**: Pathfinding in robotics, AI-based games, and optimization problems.


