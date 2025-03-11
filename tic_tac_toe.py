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


def generate_training_data(num_games, context_window):
    data: List[TrainingExample] = []
    wins_as_first = 0
    losses_as_first = 0
    draws_as_first = 0
    wins_as_second = 0
    losses_as_second = 0
    draws_as_second = 0
    report_interval = num_games // 100  # Report every 1%
    if report_interval == 0:
        report_interval = 1

    for game_num in range(num_games):
        if game_num % report_interval == 0:
            print(
                f"Generating game {game_num}/{num_games} - Wins (First): {wins_as_first}, Losses (First): {losses_as_first}, Draws (First): {draws_as_first}, Wins (Second): {wins_as_second}, Losses (Second): {losses_as_second}, Draws (Second): {draws_as_second}"
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
                if winner == first_player:
                    if first_player == 1:
                        wins_as_first += 1
                    else:
                        wins_as_second += 1
                elif winner != 0:
                    if first_player == 1:
                        losses_as_first += 1
                    else:
                        losses_as_second += 1
                else:
                    if first_player == 1:
                        draws_as_first += 1
                    else:
                        draws_as_second += 1

                for i, (row, col, move_player) in enumerate(game_moves):
                    # Assign reward to each move in the game
                    # For simplicity, we'll just use the end-of-game reward
                    # In a more advanced setup, you might want to assign
                    # intermediate rewards based on board state
                    if move_player == 1:
                        if winner == 1:
                            data.append(
                                TrainingExample(
                                    board_history=np.array(board_history),
                                    move=row * 3 + col,
                                    reward=1 if first_player == 1 else -1,
                                )
                            )
                        elif winner == 2:
                            data.append(
                                TrainingExample(
                                    board_history=np.array(board_history),
                                    move=row * 3 + col,
                                    reward=-1 if first_player == 1 else 1,
                                )
                            )
                        else:
                            data.append(
                                TrainingExample(
                                    board_history=np.array(board_history),
                                    move=row * 3 + col,
                                    reward=0,
                                )
                            )

                    elif move_player == 2:
                        if winner == 2:
                            data.append(
                                TrainingExample(
                                    board_history=np.array(board_history),
                                    move=row * 3 + col,
                                    reward=1 if first_player == 2 else -1,
                                )
                            )
                        elif winner == 1:
                            data.append(
                                TrainingExample(
                                    board_history=np.array(board_history),
                                    move=row * 3 + col,
                                    reward=-1 if first_player == 2 else 1,
                                )
                            )
                        else:
                            data.append(
                                TrainingExample(
                                    board_history=np.array(board_history),
                                    move=row * 3 + col,
                                    reward=0,
                                )
                            )

                break
    print(
        f"Finished generating data. Total Wins (First): {wins_as_first}, Losses (First): {losses_as_first}, Draws (First): {draws_as_first}, Total Wins (Second): {wins_as_second}, Losses (Second): {losses_as_second}, Draws (Second): {draws_as_second}"
    )
    total_games_first = wins_as_first + losses_as_first + draws_as_first
    if total_games_first > 0:
        print(
            f"Win percentage (First): {wins_as_first / total_games_first * 100:.2f}%"
        )
        print(
            f"Loss percentage (First): {losses_as_first / total_games_first * 100:.2f}%"
        )
        print(
            f"Draw percentage (First): {draws_as_first / total_games_first * 100:.2f}%"
        )
    total_games_second = wins_as_second + losses_as_second + draws_as_second
    if total_games_second > 0:
        print(
            f"Win percentage (Second): {wins_as_second / total_games_second * 100:.2f}%"
        )
        print(
            f"Loss percentage (Second): {losses_as_second / total_games_second * 100:.2f}%"
        )
        print(
            f"Draw percentage (Second): {draws_as_second / total_games_second * 100:.2f}%"
        )
    return data


def create_transformer_model(
    context_window=8, embedding_dim=16, num_heads=2, ff_dim=32
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

    # Output layer
    outputs = layers.Dense(10)(x)  # No softmax here

    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


class TrainingProgressCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(
            f"Epoch {epoch+1} - loss: {logs['loss']:.4f}, accuracy: {logs['accuracy']:.4f}"
        )


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

        mse = keras.losses.MeanSquaredError()
        reward_loss = mse(reward_true, reward_pred)

        # Combine losses
        total_loss = 0.9 * move_loss + 0.1 * reward_loss
        return total_loss

    model.compile(optimizer="adam", loss=custom_loss, metrics=["accuracy"])
    progress_callback = TrainingProgressCallback()
    model.fit(
        X,
        y,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[progress_callback],
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
        "--epochs", type=int, default=10, help="Number of training epochs"
    )
    train_parser.add_argument(
        "--batch_size", type=int, default=64, help="Batch size for training"
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
        default="trained_model",
        help="Output file for trained model",
    )

    # Play Game Subparser
    play_parser = subparsers.add_parser("play", help="Play a game")
    play_parser.add_argument(
        "--model_file",
        type=str,
        default="trained_model",
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
