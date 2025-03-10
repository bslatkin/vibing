import numpy as np
import random
from tensorflow import keras
from tensorflow.keras import layers
from dataclasses import dataclass


@dataclass
class TrainingExample:
    board_history: np.ndarray
    move: int
    reward: int


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

    def get_reward(self):
        """A simple reward function for Tic-Tac-Toe."""
        winner = self.check_winner()
        if winner == 1:
            return 1  # Player 1 (X) win
        elif winner == 2:
            return -1  # Player 2 (O) win
        elif self.is_board_full():
            return 0  # Draw
        else:
            return 0  # No reward yet


def generate_training_data(num_games, context_window):
    data = []
    for _ in range(num_games):
        game = TicTacToe()
        board_history = []
        game_moves = []  # Store moves made in the game
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
            game_moves.append((row, col))

            # Update history
            board_history.append(game.get_board_state().flatten())
            board_history = board_history[
                -context_window:
            ]  # Keep only context_window

            # Pad board history for uniform input
            while len(board_history) < context_window:
                board_history.insert(
                    0,
                    np.zeros(9),
                )  # Pad with empty boards at the start

            winner = game.check_winner()
            if winner != 0 or game.is_board_full():
                reward = game.get_reward()
                for i, (row, col) in enumerate(game_moves):
                    # Assign reward to each move in the game
                    # For simplicity, we'll just use the end-of-game reward
                    # In a more advanced setup, you might want to assign
                    # intermediate rewards based on board state
                    if game.current_player == 2:
                        data.append(
                            TrainingExample(
                                board_history=np.array(
                                    board_history[: len(game_moves) - i]
                                ),
                                move=row * 3 + col,
                                reward=reward,
                            )
                        )
                    else:
                        data.append(
                            TrainingExample(
                                board_history=np.array(
                                    board_history[: len(game_moves) - i]
                                ),
                                move=row * 3 + col,
                                reward=-reward,
                            )
                        )
                break
    return data


def create_transformer_model(
    context_window=8,
    embedding_dim=16,
    num_heads=2,
    ff_dim=32,
):
    """Creates a Transformer model for Tic-Tac-Toe with a context window."""

    inputs = keras.Input(shape=(context_window, 9))  # Sequence of board states

    # Embedding layer
    x = layers.Dense(embedding_dim)(inputs)

    # Reshape for multi-head attention
    x = layers.Reshape((context_window, embedding_dim))(x)

    # Transformer encoder layers
    for _ in range(2):
        # Multi-Head Self-Attention
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embedding_dim,
        )(x, x)
        x = layers.Add()([x, attention_output])  # Skip connection
        x = layers.LayerNormalization(epsilon=1e-6)(x)

        # Feed Forward
        ffn = keras.Sequential(
            [
                layers.Dense(ff_dim, activation="relu"),
                layers.Dense(embedding_dim),
            ]
        )
        ffn_output = ffn(x)
        x = layers.Add()([x, ffn_output])  # Skip connection
        x = layers.LayerNormalization(epsilon=1e-6)(x)

    # Take the last state in the sequence (current state)
    x = x[:, -1, :]

    # Output layer
    outputs = layers.Dense(9)(x)  # No softmax here

    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


def train_model(model, data, epochs=10, batch_size=32):
    X = np.array([example.board_history for example in data])
    y_move = np.array([example.move for example in data])
    y_reward = np.array([example.reward for example in data])

    # Combine move and reward into a single output
    y = np.column_stack((y_move, y_reward))

    # Custom loss function
    def custom_loss(y_true, y_pred):
        move_true = y_true[:, 0]
        reward_true = y_true[:, 1]
        move_pred = keras.activations.softmax(y_pred[:, :9])
        reward_pred = y_pred[:, 9]

        move_loss = keras.losses.sparse_categorical_crossentropy(
            move_true, move_pred
        )
        reward_loss = keras.losses.mean_squared_error(reward_true, reward_pred)

        # Combine losses (you can adjust the weights)
        total_loss = move_loss + reward_loss
        return total_loss

    model.compile(
        optimizer="adam",
        loss=custom_loss,
        metrics=["accuracy"],
    )
    model.fit(X, y, epochs=epochs, batch_size=batch_size)


def predict_next_move(model, board_history):
    """Predicts the next move based on the board history."""

    # Pad history
    while len(board_history) < model.input_shape[1]:
        board_history.insert(0, np.zeros(9))

    board_history = board_history[
        -model.input_shape[1] :
    ]  # trim the history if needed

    board_history_array = np.array(board_history).reshape(
        1,
        model.input_shape[1],
        model.input_shape[2],
    )
    predictions = model.predict(board_history_array)[0]

    # Mask out invalid moves
    current_board = board_history[-1].reshape((3, 3))
    valid_moves = np.where(current_board.flatten() == 0)[0]
    if len(valid_moves) == 0:
        return None

    # Get the move predictions
    move_predictions = keras.activations.softmax(predictions[:9]).numpy()

    best_move_index = valid_moves[np.argmax(move_predictions[valid_moves])]

    row = best_move_index // 3
    col = best_move_index % 3
    return (row, col)


def play_game(model):
    game = TicTacToe()
    board_history = []
    while True:
        print("\nCurrent Board:")
        print(game.board)

        if game.current_player == 1:
            # Model's turn
            board_history.append(game.get_board_state().flatten())
            predicted_move = predict_next_move(model, board_history)
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
                        board_history.append(game.get_board_state().flatten())
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
    training_data = generate_training_data(num_games=50000, context_window=5)

    # 2. Create Model
    print("Creating model...")
    model = create_transformer_model(context_window=5)

    # 3. Train Model
    print("Training model...")
    train_model(model, training_data, epochs=10, batch_size=64)

    # 4. Test the model
    print("\nTesting the model...")
    play_game(model)
