import time
import random
from copy import deepcopy

class TicTacToe:
    def __init__(self):
        self.board = [[' ' for _ in range(3)] for _ in range(3)]
        self.current_player = 'X'  # X starts 
    
    def print_board(self):
        print("\n")
        for i, row in enumerate(self.board):
            print(" | ".join(row))
            if i < 2:
                print("-" * 9)
        print("\n")
    
    def is_board_full(self):
        # tie
        return all(self.board[i][j] != ' ' for i in range(3) for j in range(3))
    
    def check_winner(self):
        # rows
        for row in self.board:
            if row[0] == row[1] == row[2] != ' ':
                return row[0]
        
        # columns
        for col in range(3):
            if self.board[0][col] == self.board[1][col] == self.board[2][col] != ' ':
                return self.board[0][col]
        
        # diagonals
        if self.board[0][0] == self.board[1][1] == self.board[2][2] != ' ':
            return self.board[0][0]
        if self.board[0][2] == self.board[1][1] == self.board[2][0] != ' ':
            return self.board[0][2]
        
        return None
    
    def is_game_over(self):
        return self.check_winner() is not None or self.is_board_full()
    
    def get_available_moves(self):
        return [(i, j) for i in range(3) for j in range(3) if self.board[i][j] == ' ']
    
    def make_move(self, move):
        row, col = move
        if self.board[row][col] != ' ':
            return False
        
        self.board[row][col] = self.current_player
        
        # Switch 
        self.current_player = 'O' if self.current_player == 'X' else 'X'
        return True

    def get_game_state(self):
        """Get current state of the game for Minimax"""
        winner = self.check_winner()
        if winner == 'X':
            return 1  # X wins (max)
        elif winner == 'O':
            return -1  # O wins (min)
        elif self.is_board_full():
            return 0  # Tie
        else:
            return None 


class MinimaxAI:
    def __init__(self, player_symbol, use_alpha_beta=False):
        self.player_symbol = player_symbol  
        self.opponent_symbol = 'O' if player_symbol == 'X' else 'X'
        self.use_alpha_beta = use_alpha_beta
        self.nodes_evaluated = 0
    
    def get_move(self, game):
        """Get the best move for the AI based on Minimax"""
        self.nodes_evaluated = 0
        
        # Make a copy of the game state for evaluation
        game_copy = deepcopy(game)
        
        # For the first move (when board is empty), prefer corners or center
        empty_count = sum(row.count(' ') for row in game_copy.board)
        if empty_count == 9:
            # Force the algorithm to evaluate nodes even for first move for comparison purposes
            pass
        
        best_score = float('-inf') if self.player_symbol == 'X' else float('inf')
        best_move = None
        
        for move in game_copy.get_available_moves():
            # Make the move on the copy
            current_player_saved = game_copy.current_player
            game_copy.current_player = self.player_symbol
            game_copy.make_move(move)
            
            # Evaluate move using Minimax
            if self.use_alpha_beta:
                score = self.alpha_beta_minimax(
                    game_copy, 
                    0, 
                    float('-inf'), 
                    float('inf'), 
                    self.player_symbol != 'X'  # True if minimizing
                )
            else:
                score = self.minimax(
                    game_copy, 
                    0, 
                    self.player_symbol != 'X'  # True if minimizing
                )
            
            # Restore the board
            game_copy.board[move[0]][move[1]] = ' '
            game_copy.current_player = current_player_saved
            
            # Update best move
            if self.player_symbol == 'X':  # maximizing
                if score > best_score:
                    best_score = score
                    best_move = move
            else:  # minimizing
                if score < best_score:
                    best_score = score
                    best_move = move
        
        return best_move
    
    def minimax(self, game, depth, is_minimizing):
        """Standard Minimax algorithm"""
        self.nodes_evaluated += 1
        
        # Check if game is over
        state = game.get_game_state()
        if state is not None:
            return state
        
        # Recursive case
        if is_minimizing:
            best_score = float('inf')
            for move in game.get_available_moves():
                # Make the move
                game.board[move[0]][move[1]] = 'O'
                game.current_player = 'X'
                
                # Recursive call
                score = self.minimax(game, depth + 1, False)
                
                # Undo the move
                game.board[move[0]][move[1]] = ' '
                game.current_player = 'O'
                
                best_score = min(score, best_score)
            return best_score
        else:
            best_score = float('-inf')
            for move in game.get_available_moves():
                # Make the move
                game.board[move[0]][move[1]] = 'X'
                game.current_player = 'O'
                
                # Recursive call
                score = self.minimax(game, depth + 1, True)
                
                # Undo the move
                game.board[move[0]][move[1]] = ' '
                game.current_player = 'X'
                
                best_score = max(score, best_score)
            return best_score
    
    def alpha_beta_minimax(self, game, depth, alpha, beta, is_minimizing):
        """Minimax with Alpha-Beta Pruning optimization"""
        self.nodes_evaluated += 1
        
        # Check if game is over
        state = game.get_game_state()
        if state is not None:
            return state
        
        # Recursive case
        if is_minimizing:
            best_score = float('inf')
            for move in game.get_available_moves():
                # Make the move
                game.board[move[0]][move[1]] = 'O'
                game.current_player = 'X'
                
                # Recursive call
                score = self.alpha_beta_minimax(game, depth + 1, alpha, beta, False)
                
                # Undo the move
                game.board[move[0]][move[1]] = ' '
                game.current_player = 'O'
                
                best_score = min(score, best_score)
                beta = min(beta, best_score)
                
                # Alpha-Beta pruning
                if beta <= alpha:
                    break
                    
            return best_score
        else:
            best_score = float('-inf')
            for move in game.get_available_moves():
                # Make the move
                game.board[move[0]][move[1]] = 'X'
                game.current_player = 'O'
                
                # Recursive call
                score = self.alpha_beta_minimax(game, depth + 1, alpha, beta, True)
                
                # Undo the move
                game.board[move[0]][move[1]] = ' '
                game.current_player = 'X'
                
                best_score = max(score, best_score)
                alpha = max(alpha, best_score)
                
                # Alpha-Beta pruning
                if beta <= alpha:
                    break
                    
            return best_score


def play_human_vs_ai():
    game = TicTacToe()
    use_alpha_beta = input("Use Alpha-Beta Pruning? (y/n): ").lower() == 'y'
    
    human_symbol = input("Do you want to play as X or O? (X goes first): ").upper()
    if human_symbol not in ['X', 'O']:
        human_symbol = 'X'
    
    ai_symbol = 'O' if human_symbol == 'X' else 'X'
    ai = MinimaxAI(ai_symbol, use_alpha_beta)
    
    print("\nGame starts! You are playing as", human_symbol)
    
    while not game.is_game_over():
        game.print_board()
        
        if game.current_player == human_symbol:
            try:
                move_input = input(f"Your turn ({human_symbol}). Enter row,col (0,2): ")
                row, col = map(int, move_input.split(','))
                if not (0 <= row <= 2 and 0 <= col <= 2) or game.board[row][col] != ' ':
                    print("Invalid move! Try again.")
                    continue
                game.make_move((row, col))
            except (ValueError, IndexError):
                print("Invalid input! Please use format: row,col e.g., 1,2")
                continue
        else:
            print(f"AI ({ai_symbol}) is thinking...")
            start_time = time.time()
            ai_move = ai.get_move(game)
            end_time = time.time()
            
            print(f"AI chooses: {ai_move}")
            print(f"Decision time: {end_time - start_time:.4f} seconds")
            print(f"Nodes evaluated: {ai.nodes_evaluated}")
            
            game.make_move(ai_move)
    
    game.print_board()
    winner = game.check_winner()
    
    if winner:
        if winner == human_symbol:
            print("Congratulations! You won!")
        else:
            print("AI wins! Better luck next time.")
    else:
        print("It's a tie!")


def compare_minimax_performance():
    print("\n===== PERFORMANCE COMPARISON =====")
    # Setup AIs
    standard_ai = MinimaxAI('X', use_alpha_beta=False)
    alpha_beta_ai = MinimaxAI('X', use_alpha_beta=True)
    
    # Test scenarios with different board configurations
    test_boards = [
        # Empty board
        [[' ' for _ in range(3)] for _ in range(3)],
        
        # One move made
        [['X', ' ', ' '],
         [' ', ' ', ' '],
         [' ', ' ', ' ']],
         
        # Two moves made
        [['X', ' ', ' '],
         [' ', 'O', ' '],
         [' ', ' ', ' ']],
         
        # Three moves made
        [['X', ' ', ' '],
         [' ', 'O', ' '],
         [' ', ' ', 'X']],
         
        # Four moves made
        [['X', 'O', ' '],
         [' ', 'O', ' '],
         [' ', ' ', 'X']],
    ]
    
    total_standard_nodes = 0
    total_alpha_beta_nodes = 0
    total_standard_time = 0
    total_alpha_beta_time = 0
    
    for i, board in enumerate(test_boards):
        game = TicTacToe()
        game.board = deepcopy(board)
        
        print(f"\nTest {i+1}: Board with {9 - sum(row.count(' ') for row in board)} moves made")
        for row in board:
            print(" | ".join(row))
            if i < len(board) - 1:
                print("-" * 9)
        
        # Standard Minimax
        start_time = time.time()
        standard_move = standard_ai.get_move(game)
        standard_time = time.time() - start_time
        standard_nodes = standard_ai.nodes_evaluated
        
        # Alpha-Beta Pruning
        start_time = time.time()
        alpha_beta_move = alpha_beta_ai.get_move(game)
        alpha_beta_time = time.time() - start_time
        alpha_beta_nodes = alpha_beta_ai.nodes_evaluated
        
        # Track totals
        total_standard_nodes += standard_nodes
        total_alpha_beta_nodes += alpha_beta_nodes
        total_standard_time += standard_time
        total_alpha_beta_time += alpha_beta_time
        
        # Results
        print(f"Standard Minimax: Move {standard_move}, Time {standard_time:.6f}s, Nodes {standard_nodes}")
        print(f"Alpha-Beta: Move {alpha_beta_move}, Time {alpha_beta_time:.6f}s, Nodes {alpha_beta_nodes}")
        
        # Avoid division by zero
        if alpha_beta_nodes > 0 and alpha_beta_time > 0:
            node_improvement = standard_nodes / max(alpha_beta_nodes, 1)
            time_improvement = standard_time / max(alpha_beta_time, 0.000001)
            print(f"Improvement: {node_improvement:.2f}x fewer nodes, {time_improvement:.2f}x faster")
        else:
            print("Improvement: Cannot calculate")
    
    print("\n===== OVERALL RESULTS =====")
    print(f"Total Standard Minimax: Nodes {total_standard_nodes}, Time {total_standard_time:.6f}s")
    print(f"Total Alpha-Beta: Nodes {total_alpha_beta_nodes}, Time {total_alpha_beta_time:.6f}s")
    
    # Avoid division by zero for overall results
    if total_alpha_beta_nodes > 0 and total_alpha_beta_time > 0:
        overall_node_improvement = total_standard_nodes / total_alpha_beta_nodes
        overall_time_improvement = total_standard_time / total_alpha_beta_time
        print(f"Overall Improvement: {overall_node_improvement:.2f}x fewer nodes, {overall_time_improvement:.2f}x faster")
    else:
        print("Overall Improvement: Cannot calculate")
    
    print("\n===== CONCLUSION =====")
    print("Alpha-Beta Pruning significantly reduces the number of nodes evaluated,")
    print("resulting in faster decision-making while maintaining optimal play.")


if __name__ == "__main__":
    print("Welcome to Tic-Tac-Toe with Minimax AI!")
    
    while True:
        print("\nSelect an option:")
        print("1. Play against AI")
        print("2. Compare Minimax vs Alpha-Beta Pruning")
        print("3. Exit")
        
        choice = input("Choice: ")
        
        if choice == '1':
            play_human_vs_ai()
        elif choice == '2':
            compare_minimax_performance()
        elif choice == '3':
            print("Thanks for playing!")
            break
        else:
            print("Invalid choice. Please try again.")