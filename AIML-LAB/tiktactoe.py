# Define the game board and player symbols
class TicTacToe:
    def __init__(self):
        self.board = [' ' for _ in range(9)]  # 3x3 board as a flat list
        self.current_player = 'X'  # X always starts
    
    def print_board(self):
        for i in range(0, 9, 3):
            print(f"{self.board[i]} | {self.board[i+1]} | {self.board[i+2]}")
            if i < 6:
                print("-" * 9)
    
    def is_winner(self, player):
        win_combinations = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],  # Rows
            [0, 3, 6], [1, 4, 7], [2, 5, 8],  # Columns
            [0, 4, 8], [2, 4, 6]             # Diagonals
        ]
        return any(all(self.board[i] == player for i in combo) for combo in win_combinations)
    
    def is_draw(self):
        return ' ' not in self.board  # No empty spaces
    
    def make_move(self, index, player):
        if self.board[index] == ' ':
            self.board[index] = player
            return True
        return False
    
    def dfs_ai(self, player):
        """DFS to evaluate the best move."""
        opponent = 'O' if player == 'X' else 'X'

        def dfs(board, player):
            if self.is_winner(opponent):  # If opponent won, this is a losing state
                return -1
            if self.is_draw():  # If it's a draw, return 0
                return 0
            
            best_score = -float('inf')  # Maximizing player (AI)
            for i in range(9):
                if board[i] == ' ':
                    board[i] = player
                    score = -dfs(board, opponent)  # Negate the opponent's score
                    board[i] = ' '  # Undo the move
                    best_score = max(best_score, score)
            return best_score

        best_move = -1
        best_value = -float('inf')
        for i in range(9):
            if self.board[i] == ' ':
                self.board[i] = player
                move_value = -dfs(self.board, opponent)  # Evaluate move
                self.board[i] = ' '  # Undo the move
                if move_value > best_value:
                    best_value = move_value
                    best_move = i
        return best_move

    def play(self):
        print("Welcome to Tic-Tac-Toe!")
        while True:
            self.print_board()
            if self.current_player == 'X':
                # Human player
                move = int(input("Enter your move (0-8): "))
            else:
                # AI using DFS
                print("AI is making its move...")
                move = self.dfs_ai(self.current_player)
            
            if self.make_move(move, self.current_player):
                if self.is_winner(self.current_player):
                    self.print_board()
                    print(f"Player {self.current_player} wins!")
                    break
                elif self.is_draw():
                    self.print_board()
                    print("It's a draw!")
                    break
                # Switch player
                self.current_player = 'O' if self.current_player == 'X' else 'X'
            else:
                print("Invalid move. Try again.")

# Start the game
if __name__ == "__main__":
    game = TicTacToe()
    game.play()
                    
