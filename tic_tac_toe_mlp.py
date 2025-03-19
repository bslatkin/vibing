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
    board_state: np.ndarray
    move_one_hot: np.ndarray
    row: int
    col: int
    reward: float
    last_player: int


class TicTacToe:
    def __init__(self):
        self.board = np.zeros((3, 3), dtype=int)
        self.last_player = -1
        self.current_player = 1  # Player X starts
        self.parent_board = None
        self.child_boards = {}  # Map (row, col) of the play to the child board
        self.reward_x = None
        self.reward_o = None

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

    def board_encoded(self):
        # The board is always in the perspective of the current player.
        copy = self.board.copy()
        if self.current_player == 1:
            copy[copy == 2] = -1
        else:
            copy[copy == 1] = -1
            copy[copy == 2] = 1
        return copy

    def depth(self):
        result = 0
        current = self.parent_board
        while current:
            result += 1
            current = current.parent_board
        return result


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


def calculate_reward(game):
    winner = game.check_winner()
    if winner == 1:
        game.reward_x = 1.0
        game.reward_o = -1.0
    elif winner == 2:
        game.reward_x = -1.0
        game.reward_o = 1.0
    elif game.is_board_full():
        game.reward_x = 0.0
        game.reward_o = 0.0
    else:
        max_x = -1.0
        max_o = -1.0
        min_x = 1.0
        min_o = 1.0

        for child in game.child_boards.values():
            assert child.reward_x is not None
            assert child.reward_o is not None

            max_x = max(max_x, child.reward_x)
            max_o = max(max_o, child.reward_o)
            min_x = min(min_x, child.reward_x)
            min_o = min(min_o, child.reward_o)

        if game.current_player == 1:
            game.reward_x = max_x
            game.reward_o = min_o
        elif game.current_player == 2:
            game.reward_x = min_x
            game.reward_o = max_o

        # target = np.array([[1, -1, 0], [0, 0, 1], [-1, 0, 0]])
        # if np.all(game.board_encoded() == target):
        #     x = list(game.child_boards.keys())
        #     y = [(c.reward_x, c.reward_o) for c in game.child_boards.values()]
        #     breakpoint()


def create_training_examples(row, col, game, data):
    if row != -1 and col != -1:
        if game.last_player == 1:
            reward = game.reward_x
        elif game.last_player == 2:
            reward = game.reward_o
        else:
            reward = 0.0

        move_one_hot = np.zeros(9)
        move_one_hot[row * 3 + col] = 1

        data.append(
            TrainingExample(
                board_state=game.parent_board.board_encoded(),
                move_one_hot=move_one_hot,
                row=row,
                col=col,
                reward=reward,
                last_player=game.last_player,
            )
        )
        if data and len(data) % 100_000 == 0:
            print(f"Generated {len(data)} examples")

    # TODO: Consider narrowing these examples so it's the children that
    # result in any positive outcome for X or O, but not for mistakes
    # that result in draws.
    for (child_row, child_col), child in game.child_boards.items():
        create_training_examples(child_row, child_col, child, data)


def iterate_games(game):
    yield game
    for child in game.child_boards.values():
        yield from iterate_games(child)


def generate_training_data():
    """
    Generates training data by enumerating all possible Tic-Tac-Toe games.
    """
    print("Generating all possible games...")

    game = TicTacToe()
    counter = [0]
    generate_all_games(game, counter)
    print(f"Finished generating data. {counter[0]} games")

    # This algorithm works one layer at a time starting with the deepest leaves
    # to ensure that all child nodes have been scored before their parent
    # nodes are scored.
    print("Scoring all possible games...")
    all_leaf_games = list(iterate_games(game))
    all_leaf_games.sort(key=lambda x: x.depth(), reverse=True)
    for current_game in all_leaf_games:
        calculate_reward(current_game)

    print("Creating training examples...")
    data = []
    create_training_examples(-1, -1, game, data)

    # This is like exploring a Q table to pick the best move at every state.
    # Moves at the same board state that have the same result value (equal
    # to the maximum) are included in the training data too.
    best_moves = {}
    board_count = {}
    for example in data:
        key = tuple(example.board_state.flatten())
        move_key = (example.row, example.col)

        found = best_moves.get(key)
        if not found:
            best_moves[key] = {move_key: example}
            board_count[key] = 1
        else:
            board_count[key] += 1
            max_reward = max(move.reward for move in found.values())
            if example.reward > max_reward:
                best_moves[key] = {move_key: example}
            elif example.reward == max_reward:
                found[move_key] = example

    result = []
    for move_dict in best_moves.values():
        for example in move_dict.values():
            key = tuple(example.board_state.flatten())
            count = board_count[key]
            for _ in range(count):
                result.append(example)

    print(f"Finished generating examples. {len(result)} examples")

    return result


def create_model() -> keras.Model:
    board_input = keras.Input(
        shape=(3, 3),
        name="board_input",
    )

    # l2_reg = regularizers.l2(0.0001)
    l2_reg = None

    x = layers.Flatten()(board_input)
    x = layers.Dense(4096, activation="tanh", kernel_regularizer=l2_reg)(x)
    x = layers.Dense(1024, activation="tanh", kernel_regularizer=l2_reg)(x)
    move_output = layers.Dense(
        9,
        activation="softmax",
        name="move_output",
    )(x)
    model = keras.Model = keras.Model(
        inputs=board_input,
        outputs=move_output,
    )
    return model


class TestAccuracyCallback(Callback):
    def __init__(self, X_board_input_test, y_move_test):
        super().__init__()
        self.X_board_input_test = X_board_input_test
        self.y_move_test = y_move_test
        self.test_set_size = len(X_board_input_test)

    def on_epoch_end(self, epoch, logs=None):
        if not self.X_board_input_test:
            return

        results = self.model.evaluate(
            self.X_board_input_test,
            self.y_move_test,
            verbose=0,
            return_dict=True,
        )

        print()
        print(f"Epoch {epoch+1}:")
        print(f"  Test set size: {self.test_set_size}")
        print(f"  Loss: {results['loss']:.4f}")
        print(f"  Accuracy: {results['accuracy']:.4f}")
        print()
        print()


def train_model(
    model,
    data,
    checkpoint_callback,
    epochs=5,
    batch_size=128,
    test_size=0.0,
):
    X_board = np.array([example.board_state for example in data])
    y_move = np.array([example.move_one_hot for example in data])

    if test_size:
        (
            X_board_train,
            X_board_test,
            y_move_train,
            y_move_test,
        ) = train_test_split(
            X_board,
            y_move,
            test_size=test_size,
            random_state=42,
        )
    else:
        X_board_train, y_move_train = X_board, y_move
        X_board_test, y_move_test = [], []

    learning_rate = 0.001
    optimizer = Adam(learning_rate=learning_rate)

    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    test_accuracy_callback = TestAccuracyCallback(
        X_board_test,
        y_move_test,
    )

    model.fit(
        X_board_train,
        y_move_train,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[test_accuracy_callback, checkpoint_callback],
    )
    return model


def predict_next_move(model, game):
    """Predicts the next move based on the current board state."""
    board_state = game.board_encoded()

    predictions = model.predict(
        np.expand_dims(board_state, axis=0),
        verbose=0,
    )
    move_probabilities = predictions.flatten()

    # Mask out invalid moves
    valid_moves = []
    for i in range(9):
        row, col = i // 3, i % 3
        if not game.is_valid_move(row, col):
            move_probabilities[i] = 0  # Set probability to 0 for invalid moves
        else:
            valid_moves.append((row, col))

    if not valid_moves:
        raise ValueError("No valid moves available")

    # If there are valid moves, choose one randomly
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


def inspect_data(data: list[TrainingExample]):
    """Inspects the generated test data and prints out a sequence of moves."""
    if not data:
        print("No data to inspect.")
        return

    selected_example = random.choice(data)

    # target = np.array([[0, -1, 0], [0, -1, 1], [1, 1, -1]])
    # target = np.array([[1, 1, -1], [0, 0, -1], [-1, 1, 0]])
    # target = np.array([[0, 1, 0], [-1, -1, 0], [-1, 1, 1]])
    # target = np.array([[1, -1, 0], [0, 0, 1], [-1, 0, 0]])
    # target = np.array([[-1, 0, 0], [0, 0, 0], [1, -1, 1]])
    # for example in data:
    #     if np.all(example.board_state == target):
    #         selected_example = example
    #         break

    all_examples = []
    for example in data:
        if np.all(example.board_state == selected_example.board_state):
            all_examples.append(example)

    for example in all_examples:
        print(f"Reward: {example.reward}")

        for i in range(9):
            row, col = i // 3, i % 3
            if example.move_one_hot[i]:
                print(f"Move:   ({row}, {col})")

        print("Board State (0=empty, 1=Me, -1=Opponent):")
        print(example.board_state)


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
