def initialize_board():
    return [[' ' for _ in range(3)] for _ in range(3)]

def print_board(board):
    print("-------------")
    for row in board:
        print("|", " | ".join(row), "|")
        print("-------------")

def is_winner(board, player):
    # Check rows, columns, and diagonals for a win
    for i in range(3):
        if all(board[i][j] == player for j in range(3)) or all(board[j][i] == player for j in range(3)):
            return True
    if all(board[i][i] == player for i in range(3)) or all(board[i][2 - i] == player for i in range(3)):
        return True
    return False

def is_draw(board):
    return all(board[i][j] != ' ' for i in range(3) for j in range(3))

def dfs(board, is_ai_turn):
    if is_winner(board, 'X'):
        return 1  # AI wins
    if is_winner(board, 'O'):
        return -1  # User wins
    if is_draw(board):
        return 0  # Draw

    best_score = float("-inf") if is_ai_turn else float("inf")

    for i in range(3):
        for j in range(3):
            if board[i][j] == ' ':  # Check empty spot
                board[i][j] = 'X' if is_ai_turn else 'O'
                score = dfs(board, not is_ai_turn)
                board[i][j] = ' '

                if is_ai_turn:
                    best_score = max(best_score, score)
                else:
                    best_score = min(best_score, score)

    return best_score

def find_best_move(board):
    best_score = float("-inf")
    best_move = None

    for i in range(3):
        for j in range(3):
            if board[i][j] == ' ':  # Check empty spot
                board[i][j] = 'X'
                score = dfs(board, False)
                board[i][j] = ' '

                if score > best_score:
                    best_score = score
                    best_move = (i, j)

    return best_move

def get_user_input(board):
    while True:
        try:
            row, col = map(int, input("Enter your move (row and column between 0 and 2): ").split())
            if 0 <= row <= 2 and 0 <= col <= 2 and board[row][col] == ' ':
                return row, col
            else:
                print("Invalid input. Try again.")
        except ValueError:
            print("Please enter valid numbers.")

def main():
    board = initialize_board()
    print_board(board)

    while True:
        # User's turn
        row, col = get_user_input(board)
        board[row][col] = 'O'
        print_board(board)

        if is_winner(board, 'O'):
            print("You win!")
            break
        if is_draw(board):
            print("It's a draw!")
            break

        # AI's turn
        print("AI is making a move...")
        ai_move = find_best_move(board)
        if ai_move:
            board[ai_move[0]][ai_move[1]] = 'X'
        print_board(board)

        if is_winner(board, 'X'):
            print("AI wins!")
            break
        if is_draw(board):
            print("It's a draw!")
            break

if __name__ == "__main__":
    main()
