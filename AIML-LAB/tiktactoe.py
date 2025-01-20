def print_board(board):
    for row in board:
        print(" | ".join(row))
    print()

def is_winner(board, player):
    # Check rows, columns, and diagonals
    for i in range(3):
        if all(board[i][j] == player for j in range(3)) or all(board[j][i] == player for j in range(3)):
            return True
    if all(board[i][i] == player for i in range(3)) or all(board[i][2 - i] == player for i in range(3)):
        return True
    return False

def is_full(board):
    return all(board[i][j] != ' ' for i in range(3) for j in range(3))

def get_empty_cells(board):
    return [(i, j) for i in range(3) for j in range(3) if board[i][j] == ' ']

def dfs(board, player):
    # Check if the current board state is terminal
    if is_winner(board, 'X'):
        return 1  # X wins
    if is_winner(board, 'O'):
        return -1  # O wins
    if is_full(board):
        return 0  # Draw

    # Explore all possible moves
    if player == 'X':
        best_score = -float('inf')
        for i, j in get_empty_cells(board):
            board[i][j] = player
            score = dfs(board, 'O')
            board[i][j] = ' '
            best_score = max(best_score, score)
        return best_score
    else:
        best_score = float('inf')
        for i, j in get_empty_cells(board):
            board[i][j] = player
            score = dfs(board, 'X')
            board[i][j] = ' '
            best_score = min(best_score, score)
        return best_score

def best_move(board, player):
    best_score = -float('inf') if player == 'X' else float('inf')
    move = None

    for i, j in get_empty_cells(board):
        board[i][j] = player
        score = dfs(board, 'O' if player == 'X' else 'X')
        board[i][j] = ' '

        if (player == 'X' and score > best_score) or (player == 'O' and score < best_score):
            best_score = score
            move = (i, j)
    return move

# Main game loop
def tic_tac_toe():
    board = [[' ' for _ in range(3)] for _ in range(3)]
    print("Welcome to Tic-Tac-Toe!")
    print("You are 'O', and the AI is 'X'.")
    print_board(board)

    while True:
        # Player move
        if not is_full(board) and not is_winner(board, 'X'):
            move = input("Enter your move (row and column, e.g., 1 1): ").strip().split()
            row, col = int(move[0]) - 1, int(move[1]) - 1
            if board[row][col] == ' ':
                board[row][col] = 'O'
            else:
                print("Cell already taken. Try again.")
                continue
            print("Your move:")
            print_board(board)

        # Check if player wins
        if is_winner(board, 'O'):
            print("You win!")
            break

        # AI move
        if not is_full(board) and not is_winner(board, 'O'):
            move = best_move(board, 'X')
            if move:
                board[move[0]][move[1]] = 'X'
            print("AI's move:")
            print_board(board)

        # Check if AI wins
        if is_winner(board, 'X'):
            print("AI wins!")
            break

        # Check for draw
        if is_full(board):
            print("It's a draw!")
            break

# Start the game
if __name__ == "__main__":
    tic_tac_toe()
