import argparse
import os
import pickle
import random
from dataclasses import dataclass

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
    reward: float


class TicTacToe:
    def __init__(self):
        self.board = np.zeros((3, 3), dtype=int)
        self.last_player = -1
        self.current_player = 1  # Player X starts
        self.parent_board = None
        self.child_boards = {}  # Map (row, col) of the play to the child board
        self.child_wins = 0
        self.child_losses = 0
        self.child_draws = 0

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
            child.last_player = 1
            child.current_player = 2
        else:
            child.last_player = 2
            child.current_player = 1

        if child.check_winner() != 0:
            child.winning_move = True

        return child

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
        """Converts a 3x3 board to a 3x3x3 one-hot tensor, from the current player's perspective."""
        one_hot_board = np.zeros((3, 3, 3), dtype=int)
        for r in range(3):
            for c in range(3):
                if self.board[r, c] == 0:
                    one_hot_board[r, c, 0] = 1
                elif self.board[r, c] == self.current_player:
                    one_hot_board[r, c, 1] = 1  # My piece
                else:
                    one_hot_board[r, c, 2] = 1  # Opponent's piece
        return one_hot_board


def probability_mean(values_iter):
    values = list(values_iter)
    total = sum(values)
    return total / len(values)


def generate_all_games(game, counter):
    winner = game.check_winner()
    if winner != 0 or game.is_board_full():
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


def count_outcomes(game):
    assert game.parent_board
    winner = game.check_winner()

    if winner == game.last_player:
        game.parent_board.child_wins += 1
    elif winner != 0:
        game.parent_board.child_losses += 1
    else:
        assert game.is_board_full()
        game.parent_board.child_draws += 1


def calculate_reward(game):
    winner = game.check_winner()

    if winner == game.last_player:
        return 1.0
    elif winner != 0:
        return -1.0
    elif game.is_board_full():
        return 0.0
    else:
        assert game.child_boards

        # If any of the children have immediate outcomes, then we know
        # the win probability of this specific move.
        denom = game.child_wins + game.child_losses + game.child_draws
        if denom:
            return game.child_wins / denom

        # Otherwise, recursively calculate the win probability of all the
        # children to decide this move's probability.
        total = 0.0
        count = 0
        for child in game.child_boards.values():
            total += calculate_reward(child)
            count += 1

        assert count
        return total / count


def create_training_examples(row, col, game, data):
    if row != -1 and col != -1:
        move_one_hot = np.zeros(9, dtype=int)
        move_one_hot[row * 3 + col] = 1

        reward = calculate_reward(game)

        data.append(
            TrainingExample(
                board_state_one_hot=game.parent_board.one_hot_board(),
                move_one_hot=move_one_hot,
                reward=reward,
            )
        )
        if data and len(data) % 100_000 == 0:
            print(f"Generated {len(data)} examples")

    for (child_row, child_col), child in game.child_boards.items():
        create_training_examples(child_row, child_col, child, data)


def iterate_leaf_games(game):
    for child in game.child_boards.values():
        if child.check_winner() != 0 or child.is_board_full():
            # Only yield leaf games that are complete
            yield child
        else:
            yield from iterate_leaf_games(child)


def generate_training_data():
    """
    Generates training data by enumerating all possible Tic-Tac-Toe games.
    """
    print("Generating all possible games...")

    game = TicTacToe()
    counter = [0]
    generate_all_games(game, counter)
    print(f"Finished generating data. {counter[0]} games")

    for child in iterate_leaf_games(game):
        count_outcomes(child)

    data = []
    create_training_examples(-1, -1, game, data)

    print(f"Finished generating examples. {len(data)} examples")

    return data


def one_hot_to_move(move_index):
    """Converts a move index to a (row, col) tuple."""
    return move_index // 3, move_index % 3


def create_model():
    pass


class TestAccuracyCallback(Callback):
    def __init__(
        self,
        X_board_input_test,
        y_reward_test,
        y_move_test,
    ):
        super().__init__()
        self.X_board_input_test = X_board_input_test
        self.y_reward_test = y_reward_test
        self.y_move_test = y_move_test
        self.test_set_size = len(X_board_input_test)

    def on_epoch_end(self, epoch, logs=None):
        # Reshape y_move_test to be flat for the move output
        reshaped_y_move_test = self.y_move_test[:, -1, :]

        results = self.model.evaluate(
            {
                "board_input": self.X_board_input_test,
            },
            {
                "reward_output": self.y_reward_test,
                "move_output": reshaped_y_move_test,
            },
            verbose=0,
            return_dict=True,
        )

        print()
        print()
        print(f"Epoch {epoch+1}:")
        print(f"  Test set size: {self.test_set_size}")
        print(f"  Loss: {results['loss']:.4f}")
        print(f"  Reward Loss: {results['reward_output_loss']:.4f}")
        print(f"  Move Loss: {results['move_output_loss']:.4f}")
        print(f"  Move Accuracy: {results['move_output_accuracy']:.4f}")
        print()
        print()


def train_model(
    model, data, checkpoint_callback, epochs=10, batch_size=32, test_size=0.01
):
    X_board = np.array()

    y_move = np.array()

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

    learning_rate = 0.001
    optimizer = Adam(learning_rate=learning_rate)

    model.compile(
        optimizer=optimizer,
        loss={
            "reward_output": "mse",
            "move_output": "categorical_crossentropy",
        },
        loss_weights={
            "reward_output": 0.5,
            "move_output": 0.5,
        },
        metrics={
            "move_output": "accuracy",
        },
    )

    test_accuracy_callback = TestAccuracyCallback(
        X_board_test,
        y_reward_test,
        y_move_test,
    )

    y_move_train = y_move_train[:, -1, :]
    y_move_test = y_move_test[:, -1, :]

    model.fit(
        {
            "board_input": X_board_train,
        },
        {
            "reward_output": y_reward_train,
            "move_output": y_move_train,
        },
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(
            {
                "board_input": X_board_test,
            },
            {
                "reward_output": y_reward_test,
                "move_output": y_move_test,
            },
        ),
        callbacks=[test_accuracy_callback, checkpoint_callback],
    )
    return model


def predict_next_move(model, game):
    """Predicts the next move based on the current board state."""
    all_examples_raw = extract_game_moves(game)

    if game.current_player == 1:
        all_examples = all_examples_raw[::2]
    else:
        all_examples = all_examples_raw[1::2]

    pad_examples(all_examples)

    predictions = model.predict(
        {
            "board_input": np.expand_dims(
                np.array(
                    [example.board_state_one_hot for example in all_examples],
                ),
                axis=0,
            ),
        },
        verbose=0,
    )
    move_probabilities = predictions[1][0]

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


class CheckpointCallback(Callback):
    def __init__(self, model_path):
        super().__init__()
        self.model_path = model_path

    def on_epoch_end(self, epoch, logs=None):
        try:
            self.model.save(self.model_path)
            print(
                f"\nModel checkpoint saved to {self.model_path} after epoch {epoch+1}"
            )
        except Exception as e:
            print(f"Error saving model checkpoint: {e}")

    def on_train_end(self, logs=None):
        print("\nTraining finished. Saving model...")
        try:
            self.model.save(self.model_path)
            print(f"Model saved to {self.model_path}")
        except Exception as e:
            print(f"Error saving model: {e}")


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


def one_hot_to_board(board_state_one_hot: np.ndarray):
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


def inspect_data(data):
    """Inspects the generated test data and prints out a move."""
    if not data:
        print("No data to inspect.")
        return

    # Select a random sequence
    selected_example_index = random.randint(0, len(data) - 1)
    selected_sequence = data[selected_example_index]

    print(f"Inspecting sequence {selected_example_index}:")
    print(f"Reward: {selected_sequence.reward}")

    i = 0
    for example in selected_sequence.examples:
        if np.all(example.move_one_hot == 0):
            continue

        i += 1
        print(f"\nExample {i} in sequence:")
        print("-" * 20)

        # Print the board state
        print("Board State (0=empty, 1=Me, 2=Opponent):")
        board_state = one_hot_to_board(example.board_state_one_hot)
        print(board_state)

        # Extract the row and column from the one-hot encoded move
        move_index = np.argmax(example.move_one_hot)
        row = move_index // 3
        col = move_index % 3

        # Print the move details
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
        "--resume_model",
        type=str,
        default=None,
        help="Model file to resume training from",
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
        if args.resume_model:
            model = load_model(os.path.join(args.data_dir, args.resume_model))
        else:
            model = create_model()

        output_path = os.path.join(args.data_dir, args.output_model)
        checkpoint_callback = CheckpointCallback(output_path)

        try:
            model = train_model(
                model,
                training_data,
                checkpoint_callback,
                epochs=args.epochs,
                batch_size=args.batch_size,
            )
        except KeyboardInterrupt:
            print("\nTraining interrupted by user.")
            exit(0)

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
