import argparse
import os
import pickle
import random, itertools
from dataclasses import dataclass
from typing import List

import numpy as np, tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import Callback


@dataclass
class TrainingExample:
    board_state_one_hot: np.ndarray
    move_one_hot: np.ndarray
    reward: float
    row: int
    col: int
    last_player: int


class TicTacToe:
    def __init__(self):
        self.board = np.zeros((3, 3), dtype=int)
        self.current_player = 1  # Player X starts
        self.parent_board = None
        self.child_boards = {}  # Map (row, col) of the play to the child board
        self.win_probability_x = None
        self.win_probability_o = None

    def is_valid_move(self, row, col):
        return 0 <= row < 3 and 0 <= col < 3 and self.board[row, col] == 0

    def make_move(self, row, col):
        assert self.is_valid_move(row, col)
        child = TicTacToe()
        child.board = self.board.copy()

        child.parent_board = self
        self.child_boards[(row, col)] = child

        child.board[row, col] = self.current_player

        if self.current_player == 1:
            child.current_player = 2
        else:
            child.current_player = 1

        return child

    def get_max_squares(self, row, col, player):
        """
        Returns the maximum number of squares a player has filled in
        relative to a given position.
        """
        max_present = 0

        # Check row
        row_present = 0
        for c in range(3):
            if self.board[row, c] == player:
                row_present += 1

        # Check column
        col_present = 0
        for r in range(3):
            if self.board[r, col] == player:
                col_present += 1

        # Check diagonal (top-left to bottom-right)
        diag_present = 0
        if row == col:
            for i in range(3):
                if self.board[i, i] == player:
                    diag_present += 1

        # Check diagonal (top-right to bottom-left)
        diag_right_present = 0
        if row + col == 2:
            for i in range(3):
                if self.board[i, 2 - i] == player:
                    diag_right_present += 1

        return (
            row_present,
            col_present,
            diag_present,
            diag_right_present,
        )

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

    def one_hot_board(self):
        """Converts a 3x3 board to a 3x3x3 one-hot tensor."""
        one_hot_board = np.zeros((3, 3, 3), dtype=int)
        for r in range(3):
            for c in range(3):
                if self.board[r, c] == 0:
                    one_hot_board[r, c, 0] = 1
                elif self.board[r, c] == 1:
                    one_hot_board[r, c, 1] = 1
                elif self.board[r, c] == 2:
                    one_hot_board[r, c, 2] = 1
        return one_hot_board


def generate_all_games(game, counter):
    winner = game.check_winner()
    if winner != 0 or game.is_board_full():
        if winner == 1:
            assert game.current_player == 2
            game.win_probability_x = 1.0
            game.win_probability_o = 0
        elif winner == 2:
            assert game.current_player == 1
            game.win_probability_x = 0
            game.win_probability_o = 1.0
        else:
            game.win_probability_x = 0.5
            game.win_probability_o = 0.5

        counter[0] += 1
        if counter[0] and counter[0] % 100_000 == 0:
            print(f"Generated {counter[0]} games")

        return

    empty_cells = [
        (r, c) for r in range(3) for c in range(3) if game.board[r, c] == 0
    ]

    for row, col in empty_cells:
        new_game = game.make_move(row, col)
        generate_all_games(new_game, counter)

    if game.win_probability_x is None and game.win_probability_o is None:
        assert game.child_boards
        win_x = 0
        win_o = 0
        count = 0

        for child in game.child_boards.values():
            if child.win_probability_x > 0.5:
                win_x += 1
                count += 1

            if child.win_probability_o > 0.5:
                win_o += 1
                count += 1

        if count:
            game.win_probability_x = win_x * 1.0 / count
            game.win_probability_o = win_o * 1.0 / count
        else:
            game.win_probability_x = 0.5
            game.win_probability_o = 0.5
    else:
        assert game.check_winner() != 0 or game.is_board_full()


def iterate_games(parent):
    for (row, col), child in parent.child_boards.items():
        if not child.child_boards:
            # Ignore training examples where there's only one more move,
            # since the reward function will always be zero anyways.
            continue

        yield row, col, child
        yield from iterate_games(child)


def create_training_example(row, col, game):
    # Create a one-hot vector for the move
    move_one_hot = np.zeros(9)
    move_one_hot[row * 3 + col] = 1

    # The current player is who is playing next, but we want
    # who played last time.
    last_player = 3 - game.current_player

    if last_player == 1:
        reward = game.win_probability_x
    elif last_player == 2:
        reward = game.win_probability_o
    else:
        assert False

    return TrainingExample(
        board_state_one_hot=game.parent_board.one_hot_board(),
        move_one_hot=move_one_hot,
        reward=reward,
        row=row,
        col=col,
        last_player=last_player,
    )


def generate_training_data():
    """
    Generates training data by enumerating all possible Tic-Tac-Toe games.
    """
    print("Generating all possible games...")

    game = TicTacToe()
    counter = [0]
    generate_all_games(game, counter)

    data = []
    for row, col, child in iterate_games(game):
        data.append(create_training_example(row, col, child))

    print(
        f"Final probabilities: x = {game.win_probability_x}, o = {game.win_probability_o}"
    )

    print(f"Finished generating data. {counter[0]} examples")
    return data


def create_model():
    """Creates a simple MLP model for Tic-Tac-Toe."""
    board_input = keras.Input(shape=(3, 3, 3), name="board_input")
    move_input = keras.Input(shape=(9,), name="move_input")

    x = layers.Flatten()(board_input)
    combined = layers.concatenate([x, move_input])

    # Interaction layers
    interaction_x = layers.Dense(256, activation="relu")(combined)
    interaction_x = layers.Dense(128, activation="relu")(interaction_x)
    interaction_x = layers.Dense(64, activation="relu")(interaction_x)

    # Reward output
    reward_output = layers.Dense(
        1,
        activation="sigmoid",
        name="reward_output",
    )(interaction_x)

    model = keras.Model(
        inputs=[board_input, move_input],
        outputs=[reward_output],
    )

    return model


def one_hot_to_move(move_index):
    """Converts a move index to a (row, col) tuple."""
    return move_index // 3, move_index % 3


class TestAccuracyCallback(Callback):
    def __init__(self, X_test, X_move_test, y_reward_test):
        super().__init__()
        self.X_test = X_test
        self.X_move_test = X_move_test
        self.y_reward_test = y_reward_test

    def on_epoch_end(self, epoch, logs=None):
        loss, reward_loss = self.model.evaluate(
            [self.X_test, self.X_move_test],
            self.y_reward_test,
            verbose=0,
        )

        print()
        print(f"Epoch {epoch+1}: Reward Loss: {reward_loss:.4f}")
        print()


def train_model(model, data, epochs=10, batch_size=32, test_size=0.01):
    X_board = np.array([example.board_state_one_hot for example in data])
    X_move = np.array([example.move_one_hot for example in data])
    y_reward = np.array([example.reward for example in data])

    (
        X_board_train,
        X_board_test,
        X_move_train,
        X_move_test,
        y_reward_train,
        y_reward_test,
    ) = train_test_split(
        X_board,
        X_move,
        y_reward,
        test_size=test_size,
        random_state=42,
    )

    model.compile(
        optimizer="adam",
        loss={"reward_output": "binary_crossentropy"},
        metrics={"reward_output": "mse"},
    )

    test_accuracy_callback = TestAccuracyCallback(
        X_board_test,
        X_move_test,
        y_reward_test,
    )

    model.fit(
        [X_board_train, X_move_train],
        y_reward_train,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[test_accuracy_callback],
    )
    return model


def predict_next_move(model, game):
    """Predicts the next move based on the current board state."""
    one_hot_board = game.one_hot_board()
    # Add a batch dimension to the input
    one_hot_board = np.expand_dims(one_hot_board, axis=0)
    move_probabilities = np.zeros((9,))
    for i in range(9):
        move_one_hot = np.zeros((1, 9))
        move_one_hot[0, i] = 1
        predictions = model.predict([one_hot_board, move_one_hot], verbose=0)
        move_probabilities[i] = predictions[0][0]
    # Mask out invalid moves
    for i in range(9):
        row, col = one_hot_to_move(i)
        if not game.is_valid_move(row, col):
            move_probabilities[i] = 0  # Set probability to 0 for invalid moves

    # Normalize probabilities to sum to 1 (if there are any valid moves)
    if np.sum(move_probabilities) > 0:
        move_probabilities /= np.sum(move_probabilities)

    # Print probabilities for all moves
    print("Move Probabilities:")
    for i in range(9):
        row, col = one_hot_to_move(i)
        if game.is_valid_move(row, col):
            print(f"  ({row}, {col}): {move_probabilities[i]:.4f}")
        else:
            print(f"  ({row}, {col}): Invalid")

    # Choose the move with the highest probability
    move_index = np.argmax(move_probabilities)
    return move_index // 3, move_index % 3


def play_game(model, human_player):
    game = TicTacToe()

    while True:
        print("\nCurrent Board:")
        print(game.board)

        if game.current_player == human_player:

            if human_player == 1:
                print("Your turn (X)")
            else:
                print("Your turn (O)")

            # Human's turn
            while True:
                try:
                    row = int(input("Enter row (0-2): "))
                    col = int(input("Enter column (0-2): "))
                    if game.is_valid_move(row, col):
                        game = game.make_move(row, col)
                        break
                    else:
                        print("Invalid move. Try again.")
                except ValueError:
                    print("Invalid input. Enter numbers between 0 and 2.")
        else:
            if game.current_player == 1:
                print("Model (X) is thinking...")
            else:
                print("Model (O) is thinking...")

            predicted_move = predict_next_move(model, game)
            row, col = predicted_move
            print(f"Model plays at: ({row}, {col})")
            game = game.make_move(row, col)

        winner = game.check_winner()
        if winner != 0:
            print(f"Player {winner} wins!")
            print(game.board)
            break
        elif game.is_board_full():
            print("It's a draw!")
            print(game.board)
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


def one_hot_to_board(board_state_one_hot):
    board_state = np.zeros((3, 3), dtype=int)
    for r in range(3):
        for c in range(3):
            if board_state_one_hot[r, c, 0] == 1:
                board_state[r, c] = 0
            elif board_state_one_hot[r, c, 1] == 1:
                board_state[r, c] = 1
            elif board_state_one_hot[r, c, 2] == 1:
                board_state[r, c] = 2
    return board_state


def inspect_data(data, num_examples=10):
    """Inspects the generated test data and prints out a set of games around a random index."""
    if not data:
        print("No data to inspect.")
        return

    random_index = random.randint(0, len(data) - 1)
    start_index = max(0, random_index - num_examples)
    end_index = min(len(data), random_index + num_examples + 1)

    print(
        f"Inspecting examples around index {random_index} (from {start_index} to {end_index}):"
    )

    for i in range(start_index, end_index):
        example = data[i]
        print(f"\nExample {i}:")
        print("-" * 20)

        # Convert one-hot to board state
        board_state = one_hot_to_board(example.board_state_one_hot)
        print("Board State (0=empty, 1=X, 2=O):")
        print(board_state)
        print("-" * 10)

        # Print the move and reward
        print(f"Player: {example.last_player}")
        print(f"Move:   ({example.row}, {example.col})")
        print(f"Reward: {example.reward}")
        print("-" * 20)


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
        "--epochs", type=int, default=3, help="Number of training epochs"
    )
    train_parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for training"
    )
    train_parser.add_argument(
        "--output_model",
        type=str,
        default="trained_model.keras",
        help="Output file for trained model",
    )

    # Inspect Data Subparser
    inspect_parser = subparsers.add_parser(
        "inspect", help="Inspect generated training data"
    )
    inspect_parser.add_argument(
        "--input_file",
        type=str,
        default="training_data.pkl",
        help="Input file for training data",
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
        "--human_player",
        type=int,
        default=1,
        choices=[1, 2],
        help="Which player is the human? 1 for X, 2 for O",
    )

    args = parser.parse_args()

    # Create data directory if it doesn't exist
    os.makedirs(args.data_dir, exist_ok=True)

    if args.command == "generate":
        print("Generating training data...")
        training_data = generate_training_data()
        output_path = os.path.join(args.data_dir, args.output_file)
        save_data(training_data, output_path)

    elif args.command == "train":
        print("Training model...")
        input_path = os.path.join(args.data_dir, args.input_file)
        training_data = load_data(input_path)
        model = create_model()
        model = train_model(
            model,
            training_data,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )
        output_path = os.path.join(args.data_dir, args.output_model)
        save_model(model, output_path)

    elif args.command == "inspect":
        print("Inspecting data...")
        input_path = os.path.join(args.data_dir, args.input_file)
        training_data = load_data(input_path)
        inspect_data(training_data)
    elif args.command == "play":
        print("Playing game...")
        model_path = os.path.join(args.data_dir, args.model_file)
        model = load_model(model_path)
        play_game(model, args.human_player)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
