import argparse
import os
import pickle
import random, itertools
from dataclasses import dataclass
from typing import List

import numpy as np, tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback


@dataclass
class TrainingExample:
    board_state_one_hot: np.ndarray
    move_one_hot: np.ndarray
    last_player: int


@dataclass
class TrainingSequence:
    examples: list[TrainingExample]
    reward: float


CONTEXT_WINDOW = 9


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


def probability_mean(values_iter):
    values = list(values_iter)
    total = sum(values)
    return total / len(values)


def generate_all_games(game, all_games):
    winner = game.check_winner()
    if winner != 0 or game.is_board_full():
        all_games.append(game)

        if all_games and len(all_games) % 100_000 == 0:
            print(f"Generated {len(all_games)} games")

        return

    empty_cells = [
        (r, c) for r in range(3) for c in range(3) if game.board[r, c] == 0
    ]

    for row, col in empty_cells:
        new_game = game.make_move(row, col)
        generate_all_games(new_game, all_games)


def pad_examples(examples):
    """
    Pads a list of TrainingExample objects with empty examples to ensure
    a fixed sequence length.
    """
    max_len = CONTEXT_WINDOW
    current_len = len(examples)

    if current_len >= max_len:
        return

    padding_needed = max_len - current_len

    for _ in range(padding_needed):
        empty_example = TrainingExample(
            board_state_one_hot=np.zeros((3, 3, 3)),
            move_one_hot=np.zeros(9),
            last_player=-1,
        )
        examples.insert(0, empty_example)


def create_training_examples(game):
    result = []

    current = game
    while current:
        parent = current.parent_board
        if not parent:
            break

        for (row, col), child in parent.child_boards.items():
            if child is current:
                break
        else:
            assert False

        move_one_hot = np.zeros(9)
        move_one_hot[row * 3 + col] = 1

        example = TrainingExample(
            board_state_one_hot=parent.one_hot_board(),
            move_one_hot=move_one_hot,
            last_player=0.0 if parent.current_player == 1 else 1.0,
        )
        result.insert(0, example)

        current = parent

    pad_examples(result)

    return TrainingSequence(
        examples=result,
        reward=1.0 if game.check_winner() == 1 else 0.0,
    )


def generate_training_data():
    """
    Generates training data by enumerating all possible Tic-Tac-Toe games.
    """
    print("Generating all possible games...")

    game = TicTacToe()
    all_games = []
    generate_all_games(game, all_games)
    print(f"Finished generating data. {len(all_games)} games")

    data = []
    for game in all_games:
        if data and len(data) % 100_000 == 0:
            print(f"Generated {len(data)} examples")
        data.append(create_training_examples(game))

    print(f"Finished generating examples. {len(data)} examples")

    return data


def create_transformer_model(
    sequence_length=CONTEXT_WINDOW,
    embedding_dim=128,
    num_heads=4,
    ff_dim=32,
    num_transformer_blocks=2,
):
    # Input: Sequence of board states
    board_input = keras.Input(
        shape=(sequence_length, 3, 3, 3), name="board_input"
    )

    # Flatten the board states
    x = layers.Reshape((sequence_length, 27))(board_input)

    # Embedding layer
    x = layers.Dense(embedding_dim, activation="relu")(x)

    # Positional encoding
    positional_encodings = np.zeros((sequence_length, embedding_dim))
    for pos in range(sequence_length):
        for i in range(0, embedding_dim, 2):
            denominator = np.power(10000, i / embedding_dim)
            positional_encodings[pos, i] = np.sin(pos / denominator)
            if i + 1 < embedding_dim:
                positional_encodings[pos, i + 1] = np.cos(pos / denominator)

    positional_encodings = tf.constant(positional_encodings, dtype=tf.float32)
    x = x + positional_encodings

    # Transformer blocks
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, embedding_dim, num_heads, ff_dim)

    # Reward branch (single output for the entire sequence)
    reward_x = layers.GlobalAveragePooling1D()(x)
    reward_x = layers.Dense(128, activation="relu")(reward_x)
    reward_output = layers.Dense(
        1, activation="sigmoid", name="reward_output"
    )(reward_x)

    # Move branch
    move_x = x[:, -1, :]  # Take the last vector in the sequence
    move_x = layers.Dense(128, activation="relu")(move_x)
    move_output = layers.Dense(
        9,
        activation="softmax",
        name="move_output",
    )(move_x)

    model = keras.Model(
        inputs={"board_input": board_input},
        outputs=[reward_output, move_output],
    )
    return model


# Helper function for a transformer block
def transformer_encoder(inputs, embedding_dim, num_heads, ff_dim):
    # Attention and Normalization
    x = layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=embedding_dim,
    )(inputs, inputs)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    res = x + inputs

    # Feed Forward and Normalization
    x = layers.Dense(ff_dim, activation="relu")(res)
    x = layers.Dense(embedding_dim)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    return x + res


def one_hot_to_move(move_index):
    """Converts a move index to a (row, col) tuple."""
    return move_index // 3, move_index % 3


class TestAccuracyCallback(Callback):
    def __init__(self, X_test, y_reward_test, y_move_test, sequence_length):
        super().__init__()
        self.X_test = X_test
        self.y_reward_test = y_reward_test
        self.y_move_test = y_move_test
        self.sequence_length = sequence_length

    def on_epoch_end(self, epoch, logs=None):
        # Reshape y_move_test to match the output shape
        y_move_test_reshaped = self.y_move_test[:, -1, :]
        loss, reward_loss, move_loss, move_accuracy, reward_mse = (
            self.model.evaluate(
                {"board_input": self.X_test},
                {
                    "reward_output": self.y_reward_test,
                    "move_output": y_move_test_reshaped,
                },
                verbose=0,
            )
        )
        print()
        print(
            f"Epoch {epoch+1}: Reward Loss: {reward_loss:.4f}, Move Loss: {move_loss:.4f}, Reward MSE: {reward_mse:.4f}, Move Accuracy: {move_accuracy:.4f}"
        )
        print()


def train_model(model, data, epochs=10, batch_size=32, test_size=0.01):
    X_board = np.array(
        [
            np.array(
                [example.board_state_one_hot for example in sequence.examples]
            )
            for sequence in data
        ]
    )
    y_move = np.array(
        [
            np.array([example.move_one_hot for example in sequence.examples])
            for sequence in data
        ]
    )
    y_reward = np.array([sequence.reward for sequence in data])
    (
        X_board_train,
        X_board_test,
        y_move_train,
        y_move_test,
        y_reward_train,
        y_reward_test,
    ) = train_test_split(
        X_board,
        y_move,
        y_reward,
        test_size=test_size,
        random_state=42,
    )

    learning_rate = 0.00005
    optimizer = Adam(learning_rate=learning_rate)

    model.compile(
        optimizer=optimizer,
        loss={
            "reward_output": "binary_crossentropy",
            "move_output": "categorical_crossentropy",
        },
        loss_weights={"reward_output": 0.5, "move_output": 0.5},
        metrics={"reward_output": "mse", "move_output": "accuracy"},
    )

    test_accuracy_callback = TestAccuracyCallback(
        X_board_test,
        y_reward_test,
        y_move_test,
        sequence_length=CONTEXT_WINDOW,
    )

    # Reshape y_move_train and y_move_test to be flat for the move output
    y_move_train = y_move_train[:, -1, :]
    y_move_test = y_move_test[:, -1, :]

    model.fit(
        {"board_input": X_board_train},
        {"reward_output": y_reward_train, "move_output": y_move_train},
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[test_accuracy_callback],
    )
    return model


def predict_next_move(model, game, optimize_for_win):
    """Predicts the next move based on the current board state."""
    one_hot_board = game.one_hot_board()
    # Add a batch dimension to the input
    # one_hot_board = np.expand_dims(one_hot_board, axis=0)

    # Create a sequence of board states for the model
    board_sequence = np.zeros((CONTEXT_WINDOW, 3, 3, 3))
    board_sequence[-1] = one_hot_board
    board_sequence = np.expand_dims(board_sequence, axis=0)

    predictions = model.predict({"board_input": board_sequence}, verbose=0)
    move_probabilities = predictions[1][0]

    if not optimize_for_win:
        # Invert the probabilities to optimize for losing
        move_probabilities = 1 - move_probabilities

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

            predicted_move = predict_next_move(model, game, human_player == 1)
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


def inspect_data(data: list[TrainingSequence]):
    """Inspects the generated test data and prints out a sequence of moves."""
    if not data:
        print("No data to inspect.")
        return

    # Select a random sequence
    selected_example_index = random.randint(0, len(data) - 1)
    selected_sequence = data[selected_example_index]

    print(f"Inspecting sequence {selected_example_index}:")
    print(f"Reward: {selected_sequence.reward}")

    for i, example in enumerate(selected_sequence.examples):
        if np.all(example.move_one_hot == 0):
            continue

        print(f"\nExample {i+1} in sequence:")
        print("-" * 20)

        # Print the board state
        print("Board State (0=empty, 1=X, 2=O):")
        board_state = one_hot_to_board(example.board_state_one_hot)
        print(board_state)

        # Extract the row and column from the one-hot encoded move
        move_index = np.argmax(example.move_one_hot)
        row = move_index // 3
        col = move_index % 3

        # Print the move details
        print(f"Player: {'O' if example.last_player == 1 else 'X'}")
        print(f"Move:   ({row}, {col})")

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
        model = create_transformer_model()
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
        optimize_for_win = True
        play_game(model, args.human_player)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
