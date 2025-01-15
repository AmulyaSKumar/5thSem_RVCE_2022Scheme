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
