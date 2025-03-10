import numpy as np
import random
from tensorflow import keras
from tensorflow.keras import layers


class TicTacToe:
    def __init__(self):
        self.board = np.zeros((3, 3), dtype=int)
        self.current_player = 1  # Player X starts

    def is_valid_move(self, row, col):
        return 0 <= row < 3 and 0 <= col < 3 and self.board[row, col] == 0

    def make_move(self, row, col):
        if self.is_valid_move(row, col):
            self.board[row, col] = self.current_player
            self.current_player = (
                3 - self.current_player
            )  # Switch players (1 -> 2, 2 -> 1)
            return True
        return False

    def check_winner(self):
        # Check rows
        for row in self.board:
            if row[0] == row[1] == row[2] and row[0] != 0:
                return row[0]
        # Check columns
        for col in range(3):
            if (
                self.board[0, col] == self.board[1, col] == self.board[2, col]
                and self.board[0, col] != 0
            ):
                return self.board[0, col]
        # Check diagonals
        if (
            self.board[0, 0] == self.board[1, 1] == self.board[2, 2]
            and self.board[0, 0] != 0
        ):
            return self.board[0, 0]
        if (
            self.board[0, 2] == self.board[1, 1] == self.board[2, 0]
            and self.board[0, 2] != 0
        ):
            return self.board[0, 2]
        return 0  # No winner

    def is_board_full(self):
        return np.all(self.board != 0)

    def reset(self):
        self.board = np.zeros((3, 3), dtype=int)
        self.current_player = 1

    def get_board_state(self):
        return self.board.copy()


def generate_training_data(num_games):
    data = []
    game = TicTacToe()
    for _ in range(num_games):
        game.reset()
        while True:
            empty_cells = [
                (r, c)
                for r in range(3)
                for c in range(3)
                if game.board[r, c] == 0
            ]
            if not empty_cells:
                break
            row, col = random.choice(empty_cells)
            game.make_move(row, col)
            data.append((game.get_board_state().flatten(), row * 3 + col))

            winner = game.check_winner()
            if winner != 0 or game.is_board_full():
                break
    return data


def create_model():
    model = keras.Sequential(
        [
            layers.Dense(128, activation="relu", input_shape=(9,)),
            layers.Dense(64, activation="relu"),
            layers.Dense(
                9, activation="softmax"
            ),  # Output layer with 9 probabilities
        ]
    )
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_model(model, data, epochs=10, batch_size=32):
    X = np.array([board for board, _ in data])
    y = np.array([move for _, move in data])
    model.fit(X, y, epochs=epochs, batch_size=batch_size)


def predict_next_move(model, board):
    board_flat = board.flatten().reshape(1, 9)
    predictions = model.predict(board_flat)[0]

    # Mask out invalid moves
    valid_moves = np.where(board.flatten() == 0)[0]
    if len(valid_moves) == 0:
        return None

    valid_predictions = predictions[valid_moves]
    best_move_index = valid_moves[np.argmax(valid_predictions)]

    row = best_move_index // 3
    col = best_move_index % 3
    return (row, col)


def play_game(model):
    game = TicTacToe()
    while True:
        print("\nCurrent Board:")
        print(game.board)

        if game.current_player == 1:
            # Model's turn
            predicted_move = predict_next_move(model, game.get_board_state())
            if predicted_move is None:
                print("No valid moves left.")
                break
            row, col = predicted_move
            print(f"Model (X) plays at: ({row}, {col})")
            game.make_move(row, col)
        else:
            # Human's turn
            while True:
                try:
                    row = int(input("Enter row (0-2): "))
                    col = int(input("Enter column (0-2): "))
                    if game.make_move(row, col):
                        break
                    else:
                        print("Invalid move. Try again.")
                except ValueError:
                    print("Invalid input. Enter numbers between 0 and 2.")

        winner = game.check_winner()
        if winner != 0:
            print(f"Player {winner} wins!")
            break
        elif game.is_board_full():
            print("It's a draw!")
            break


# --- Main Execution ---
if __name__ == "__main__":
    # 1. Generate Data
    print("Generating training data...")
    training_data = generate_training_data(num_games=50000)

    # 2. Create Model
    print("Creating model...")
    model = create_model()

    # 3. Train Model
    print("Training model...")
    train_model(model, training_data, epochs=10, batch_size=64)

    # 4. Test the model
    print("\nTesting the model...")
    play_game(model)
