import argparse
import os
import pickle
import random
from dataclasses import dataclass
from typing import List

import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import Callback


@dataclass
class TrainingExample:
    board_history: np.ndarray
    move: int
    reward: int


class TicTacToe:
    def __init__(self):
        self.board = np.zeros((3, 3), dtype=int)
        self.current_player = 1  # Player X starts
        self.move_history = []
        self.board_history = []

    def is_valid_move(self, row, col):
        return 0 <= row < 3 and 0 <= col < 3 and self.board[row, col] == 0

    def make_move(self, row, col):
        if self.is_valid_move(row, col):
            self.board[row, col] = self.current_player
            self.current_player = (
                3 - self.current_player
            )  # Switch players (1 -> 2, 2 -> 1)
            self.move_history.append((row, col, self.current_player))
            self.board_history.append(self.get_board_state().flatten())
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
        self.move_history = []
        self.board_history = []

    def get_board_state(self):
        return self.board.copy()


class GameStats:
    def __init__(self, name):
        self.name = name
        self.wins = 0
        self.losses = 0
        self.draws = 0

    def add_win(self):
        self.wins += 1

    def add_loss(self):
        self.losses += 1

    def add_draw(self):
        self.draws += 1

    def __str__(self):
        total_games = self.wins + self.losses + self.draws
        win_percentage = (
            (self.wins / total_games * 100) if total_games > 0 else 0
        )
        return f"Total Wins ({self.name}): {self.wins}, Losses ({self.name}): {self.losses}, Draws ({self.name}): {self.draws}, Win percentage ({self.name}): {win_percentage:.2f}%"


def calculate_reward(game, row, col, move_player):
    """Calculates the reward for a move."""
    if move_player != 2:
        return 0
    reward = 0

    # Check if the move wins the game
    temp_game = TicTacToe()
    temp_game.board = game.board.copy()
    temp_game.current_player = game.current_player
    temp_game.make_move(row, col)
    if temp_game.check_winner() == 2:
        reward = 1

    # Check if the move creates two in a row for player 2
    for r in range(3):
        for c in range(3):
            if temp_game.board[r, c] == 2:
                for dr, dc in [(0, 1), (1, 0), (1, 1), (1, -1)]:
                    if (
                        0 <= r + dr < 3
                        and 0 <= c + dc < 3
                        and temp_game.board[r + dr, c + dc] == 2
                    ):
                        reward = 1

    # Check if the move blocks player 1 from winning
    for r in range(3):
        for c in range(3):
            if temp_game.board[r, c] == 0:
                temp_game.board[r, c] = 1
                if temp_game.check_winner() == 1:
                    reward = 1
                temp_game.board[r, c] = 0

    # Check if the move allows player 1 to win in the next turn
    for r in range(3):
        for c in range(3):
            if temp_game.board[r, c] == 0:
                temp_game.board[r, c] = 1
                if temp_game.check_winner() == 1:
                    reward = -1
                temp_game.board[r, c] = 0

    # Check if the move allows player 1 to get two in a row in the next turn
    for r in range(3):
        for c in range(3):
            if temp_game.board[r, c] == 1:
                for dr, dc in [(0, 1), (1, 0), (1, 1), (1, -1)]:
                    if (
                        0 <= r + dr < 3
                        and 0 <= c + dc < 3
                        and temp_game.board[r + dr, c + dc] == 1
                    ):
                        reward = -1

    return reward


def create_training_examples(game, winner, context_window):
    """Creates TrainingExample instances from a completed game."""
    data: List[TrainingExample] = []
    for i, (row, col, move_player) in enumerate(game.move_history):
        if move_player == 2:
            padded_history = [np.zeros(9)] * (
                context_window - min(context_window, i + 1)
            ) + [x.copy() for x in game.board_history[: i + 1]]
            padded_board_history = [
                x.copy() for x in padded_history[-context_window:]
            ]

            reward = calculate_reward(game, row, col, move_player)
            data.append(
                TrainingExample(
                    board_history=np.array(padded_board_history),
                    move=row * 3 + col,
                    reward=reward,
                )
            )
    return data


def generate_training_data(num_games, context_window):
    data: List[TrainingExample] = []
    report_interval = num_games // 100  # Report every 1%
    if report_interval == 0:
        report_interval = 1

    second_player_stats = GameStats("Second")

    for game_num in range(num_games):
        if game_num % report_interval == 0:
            print(
                f"Generating game {game_num}/{num_games} - {second_player_stats}"
            )
        game = TicTacToe()

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
            if game.make_move(row, col):
                winner = game.check_winner()
                if winner != 0 or game.is_board_full():
                    if winner == 2:
                        second_player_stats.add_win()
                    elif winner == 1:
                        second_player_stats.add_loss()
                    else:
                        second_player_stats.add_draw()

                    data.extend(
                        create_training_examples(game, winner, context_window)
                    )

                    break
    print("Finished generating data.")

    print(second_player_stats)
    return data


def create_transformer_model(
    context_window=5, embedding_dim=16, num_heads=2, ff_dim=32
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
            num_heads=num_heads, key_dim=embedding_dim
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

    move_outputs = layers.Dense(9, activation="softmax", name="move_output")(x)
    reward_outputs = layers.Dense(1, activation="tanh", name="reward_output")(
        x
    )
    model = keras.Model(inputs=inputs, outputs=[move_outputs, reward_outputs])
    return model


class TestAccuracyCallback(Callback):
    def __init__(self, X_test, y_move_test, y_reward_test):
        super().__init__()
        self.X_test = X_test
        self.y_move_test = y_move_test
        self.y_reward_test = y_reward_test

    def on_epoch_end(self, epoch, logs=None):
        loss, move_loss, reward_loss, move_accuracy = self.model.evaluate(
            self.X_test,
            {
                "move_output": self.y_move_test,
                "reward_output": self.y_reward_test,
            },
            verbose=0,
        )
        print()
        print(
            f"Epoch {epoch+1}: Move Accuracy: {move_accuracy:.4f}, Reward Loss: {reward_loss:.4f}"
        )
        print()


def train_model(model, data, epochs=10, batch_size=32, test_size=0.05):
    X = np.array([example.board_history for example in data])
    y_move = np.array([example.move for example in data])
    y_reward = np.array([example.reward for example in data])

    (
        X_train,
        X_test,
        y_move_train,
        y_move_test,
        y_reward_train,
        y_reward_test,
    ) = train_test_split(
        X, y_move, y_reward, test_size=test_size, random_state=42
    )

    model.compile(
        optimizer="adam",
        loss={
            "move_output": "sparse_categorical_crossentropy",
            "reward_output": "mse",
        },
        loss_weights={"move_output": 0.5, "reward_output": 0.5},
        metrics={"move_output": "accuracy"},
    )

    test_accuracy_callback = TestAccuracyCallback(
        X_test, y_move_test, y_reward_test
    )

    model.fit(
        X_train,
        {"move_output": y_move_train, "reward_output": y_reward_train},
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[test_accuracy_callback],
    )
    return model


def predict_next_move(model, board_history):
    """Predicts the next move based on the board history."""

    # Pad history
    while len(board_history) < model.input_shape[1]:
        board_history.insert(0, np.zeros(9))

    board_history = board_history[
        -model.input_shape[1] :
    ]  # trim the history if needed

    board_history_array = np.array(board_history).reshape(
        1, model.input_shape[1], model.input_shape[2]
    )
    predictions = model.predict(board_history_array, verbose=0)
    # Corrected part:
    move_predictions = predictions[0][0]

    # Mask out invalid moves
    current_board = board_history[-1].reshape((3, 3))
    valid_moves = np.where(current_board.flatten() == 0)[0]

    # Handle no valid moves
    if len(valid_moves) == 0:
        return None

    # Handle only one valid move
    if len(valid_moves) == 1:
        best_move_index = valid_moves[0]
    else:
        # Get the move predictions
        move_predictions = keras.activations.softmax(move_predictions).numpy()

        # Corrected part:
        best_move_index_in_valid_moves = np.argmax(
            move_predictions[valid_moves]
        )
        best_move_index = valid_moves[best_move_index_in_valid_moves]

    row = best_move_index // 3
    col = best_move_index % 3
    return (row, col)


def play_game(model, context_window):
    game = TicTacToe()
    board_history = [np.zeros(9)] * context_window
    while True:
        print("\nCurrent Board:")
        print(game.board)

        if game.current_player == 2:
            board_history.append(game.get_board_state().flatten())
            predicted_move = predict_next_move(
                model, board_history[-context_window:]
            )
            row, col = predicted_move
            print(f"Model (O) plays at: ({row}, {col})")
            game.make_move(row, col)

        elif game.current_player == 1:
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


def save_data(data, filename):
    with open(filename, "wb") as f:
        pickle.dump(data, f)
    print(f"Data saved to {filename}")


def load_data(filename):
    with open(filename, "rb") as f:
        data = pickle.load(f)
    print(f"Data loaded from {filename}")
    return data


def save_model(model, filename):
    model.save(filename)
    print(f"Model saved to {filename}")


def load_model(filename):
    model = keras.models.load_model(filename)
    print(f"Model loaded from {filename}")
    return model


def main():
    parser = argparse.ArgumentParser(description="Tic-Tac-Toe AI")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Directory to store data and models",
    )
    subparsers = parser.add_subparsers(dest="command", help="Sub-command help")

    # Generate Data Subparser
    generate_parser = subparsers.add_parser(
        "generate", help="Generate training data"
    )
    generate_parser.add_argument(
        "--num_games",
        type=int,
        default=50_000,
        help="Number of games to generate",
    )
    generate_parser.add_argument(
        "--context_window",
        type=int,
        default=5,
        help="Context window size for board history",
    )
    generate_parser.add_argument(
        "--output_file",
        type=str,
        default="training_data.pkl",
        help="Output file for training data",
    )

    # Train Model Subparser
    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument(
        "--input_file",
        type=str,
        default="training_data.pkl",
        help="Input file for training data",
    )
    train_parser.add_argument(
        "--epochs", type=int, default=10, help="Number of training epochs"
    )
    train_parser.add_argument(
        "--batch_size", type=int, default=1024, help="Batch size for training"
    )
    train_parser.add_argument(
        "--context_window",
        type=int,
        default=5,
        help="Context window size for board history",
    )
    train_parser.add_argument(
        "--output_model",
        type=str,
        default="trained_model.keras",
        help="Output file for trained model",
    )

    # Play Game Subparser
    play_parser = subparsers.add_parser("play", help="Play a game")
    play_parser.add_argument(
        "--model_file",
        type=str,
        default="trained_model.keras",
        help="Model file to use for playing",
    )
    play_parser.add_argument(
        "--context_window",
        type=int,
        default=5,
        help="Context window size for board history",
    )

    args = parser.parse_args()

    # Create data directory if it doesn't exist
    os.makedirs(args.data_dir, exist_ok=True)

    if args.command == "generate":
        print("Generating training data...")
        training_data = generate_training_data(
            args.num_games, args.context_window
        )
        output_path = os.path.join(args.data_dir, args.output_file)
        save_data(training_data, output_path)

    elif args.command == "train":
        print("Training model...")
        input_path = os.path.join(args.data_dir, args.input_file)
        training_data = load_data(input_path)
        model = create_transformer_model(context_window=args.context_window)
        model = train_model(
            model,
            training_data,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )
        output_path = os.path.join(args.data_dir, args.output_model)
        save_model(model, output_path)

    elif args.command == "play":
        print("Playing game...")
        model_path = os.path.join(args.data_dir, args.model_file)
        model = load_model(model_path)
        play_game(model, args.context_window)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
