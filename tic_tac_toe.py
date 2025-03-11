import argparse
import os
import pickle
import random
from dataclasses import dataclass
from typing import List

import numpy as np
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


def generate_training_data(num_games, context_window):
    data: List[TrainingExample] = []
    report_interval = num_games // 100  # Report every 1%
    if report_interval == 0:
        report_interval = 1

    first_player_stats = GameStats("First")
    second_player_stats = GameStats("Second")

    for game_num in range(num_games):
        if game_num % report_interval == 0:
            print(
                f"Generating game {game_num}/{num_games} - {first_player_stats}, {second_player_stats}"
            )
        game = TicTacToe()
        board_history = []
        game_moves = []  # Store moves made in the game
        first_player = random.choice([1, 2])
        current_player = first_player
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
            game_moves.append((row, col, current_player))
            current_player = 3 - current_player

            # Update history
            board_state = game.get_board_state().flatten()
            board_history.append(board_state)

            winner = game.check_winner()
            if winner != 0 or game.is_board_full():
                reward = game.get_reward()

                if winner == first_player:
                    (
                        first_player_stats
                        if first_player == 1
                        else second_player_stats
                    ).add_win()
                elif winner != 0:
                    (
                        first_player_stats
                        if first_player == 1
                        else second_player_stats
                    ).add_loss()
                else:
                    (
                        first_player_stats
                        if first_player == 1
                        else second_player_stats
                    ).add_draw()

                for i, (row, col, move_player) in enumerate(game_moves):
                    # Pad board history for uniform input
                    padded_history = [np.zeros(9)] * (
                        context_window - min(context_window, i + 1)
                    ) + board_history[: i + 1]
                    padded_board_history = padded_history[-context_window:]

                    # Assign reward to each move based on the outcome
                    reward = 0
                    if winner != 0:
                        if winner == move_player:
                            reward = 1
                        else:
                            reward = -1
                    else:
                        reward = 0

                    # Adjust reward based on who the first player was
                    if first_player != move_player:
                        reward *= -1

                    data.append(
                        TrainingExample(
                            board_history=np.array(padded_board_history),
                            move=row * 3 + col,
                            reward=reward,
                        )
                    )

                break
    print(f"Finished generating data.")
    print(first_player_stats)
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


def train_model(model, data, epochs=10, batch_size=32):
    X = np.array([example.board_history for example in data])
    y_move = np.array([example.move for example in data])
    y_reward = np.array([example.reward for example in data])

    model.compile(
        optimizer="adam",
        loss={
            "move_output": "sparse_categorical_crossentropy",
            "reward_output": "mse",
        },
        loss_weights={"move_output": 0.7, "reward_output": 0.3},
        metrics={"move_output": "accuracy"},
    )
    y_move = np.array([example.move for example in data])
    y_reward = np.array([example.reward for example in data])
    y = np.column_stack((y_move, y_reward))
    model.fit(
        X,
        {"move_output": y_move, "reward_output": y_reward},
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[],
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
        default=50000,
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
        "--epochs", type=int, default=2, help="Number of training epochs"
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
        play_game(model)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
