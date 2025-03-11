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
    board_state: np.ndarray
    move_one_hot: np.ndarray
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
            # Save the current state of the board
            self.move_history.append((row, col, self.current_player))
            self.board_history.append(self.get_board_state().flatten())

            # Update the state of the board based on the move
            self.board[row, col] = self.current_player
            self.current_player = (
                3 - self.current_player
            )  # Switch players (1 -> 2, 2 -> 1)
            return True
        else:
            return False

    def count_touching_squares(self, player):
        """Counts the maximum number of touching squares for a given player."""
        max_touching = 0
        for row in range(3):
            for col in range(3):
                if self.board[row, col] == player:
                    touching = 0
                    # Check adjacent squares
                    for dr, dc in [
                        (-1, 0),
                        (1, 0),
                        (0, -1),
                        (0, 1),
                        (-1, -1),
                        (-1, 1),
                        (1, -1),
                        (1, 1),
                    ]:
                        nr, nc = row + dr, col + dc
                        if (
                            0 <= nr < 3
                            and 0 <= nc < 3
                            and self.board[nr, nc] == player
                        ):
                            touching += 1
                    max_touching = max(max_touching, touching)
        return max_touching

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

    def copy(self):
        new_game = TicTacToe()
        new_game.board = self.board.copy()
        new_game.current_player = self.current_player
        new_game.move_history = self.move_history[:]
        new_game.board_history = self.board_history[:]
        return new_game

    def copy_until_move(self, move_index):
        new_game = TicTacToe()
        for row, col, _ in self.move_history[:move_index]:
            new_game.make_move(row, col)
        return new_game


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


def calculate_reward(game, move_index):
    """Calculates the reward for a move at a specific index in the game history."""
    before_game = game.copy_until_move(move_index)
    after_game = game.copy_until_move(move_index + 1)

    player = before_game.current_player
    opponent = after_game.current_player

    player_wins = False
    player_more_touching = False
    opponent_will_win = False
    opponent_more_touching = False

    if after_game.check_winner() == player:
        # Current player wins on this turn
        player_wins = True

    # Check if the opponent can win on their next turn
    opponent_next_moves = [
        (r, c)
        for r in range(3)
        for c in range(3)
        if after_game.board[r, c] == 0
    ]
    for next_row, next_col in opponent_next_moves:
        temp_game = after_game.copy()
        temp_game.make_move(next_row, next_col)

        if temp_game.check_winner() == opponent:
            opponent_can_win = True

        before_count = after_game.count_touching_squares(opponent)
        after_count = temp_game.count_touching_squares(opponent)
        if before_count and after_count > before_count:
            opponent_more_touching = True

    # Player increases touching on this turn
    before_count = before_game.count_touching_squares(player)
    after_count = after_game.count_touching_squares(player)
    if before_count and after_count > before_count:
        player_more_touching = True

    winner = game.check_winner()

    if player_wins:
        return 1
    elif opponent_will_win:
        return -1
    # elif player_more_touching:
    #     return 0.5
    # elif opponent_more_touching:
    #     return -0.5
    elif player == winner:
        return 0.1
    elif opponent == winner:
        return -0.1
    else:
        return 0


def create_training_examples(game, data):
    """Creates TrainingExample instances from a completed game."""
    for move_index, (row, col, player) in enumerate(game.move_history):
        # Get the board state before this move
        board_state = game.board_history[move_index]

        # Create a one-hot vector for the move
        move_one_hot = np.zeros(9)
        move_one_hot[row * 3 + col] = 1

        # Calculate reward relative to the player who made the move
        reward = calculate_reward(game, move_index)
        data.append(
            TrainingExample(
                board_state=board_state,
                move_one_hot=move_one_hot,
                reward=reward,
            )
        )

        if data and len(data) % 100_000 == 0:
            print(f"Generated {len(data)} data points...")


def generate_all_games(game, data, player_stats):
    """
    Recursively generates all possible Tic-Tac-Toe games and extracts training data.
    """
    winner = game.check_winner()
    if winner != 0 or game.is_board_full():
        if winner == 2:
            player_stats[0].add_loss()
            player_stats[1].add_win()
        elif winner == 1:
            player_stats[0].add_win()
            player_stats[1].add_loss()
        else:
            player_stats[0].add_draw()
            player_stats[1].add_draw()

        create_training_examples(game, data)
        return

    empty_cells = [
        (r, c) for r in range(3) for c in range(3) if game.board[r, c] == 0
    ]

    for row, col in empty_cells:
        new_game = game.copy()
        assert new_game.make_move(row, col)
        generate_all_games(new_game, data, player_stats)


def generate_training_data():
    """
    Generates training data by enumerating all possible Tic-Tac-Toe games.
    """
    print("Generating all possible games...")

    game = TicTacToe()
    data = []
    player_stats = [GameStats("First"), GameStats("Second")]
    generate_all_games(game, data, player_stats)
    print("Finished generating data.")
    print(player_stats[0])
    print(player_stats[1])
    return data


def create_model():
    """Creates a simple neural network model for Tic-Tac-Toe."""
    board_input = keras.Input(shape=(9,), name="board_input")
    x = layers.Dense(64, activation="relu")(board_input)
    x = layers.Dense(32, activation="relu")(x)

    move_output = layers.Dense(9, activation="sigmoid", name="move_output")(x)
    reward_output = layers.Dense(1, activation="tanh", name="reward_output")(x)

    model = keras.Model(
        inputs=board_input, outputs=[move_output, reward_output]
    )

    return model


def one_hot_to_move(move_index):
    """Converts a move index to a (row, col) tuple."""
    return move_index // 3, move_index % 3


class TestAccuracyCallback(Callback):
    def __init__(self, X_test, y_move_test, y_reward_test):
        super().__init__()
        self.X_test = X_test
        self.y_move_test = y_move_test
        self.y_reward_test = y_reward_test

    def on_epoch_end(self, epoch, logs=None):
        loss, move_loss, reward_loss, move_accuracy, reward_mse = (
            self.model.evaluate(
                self.X_test,
                {
                    "move_output": self.y_move_test,
                    "reward_output": self.y_reward_test,
                },
                verbose=0,
            )
        )

        print()
        print(
            f"Epoch {epoch+1}: Move Accuracy: {move_accuracy:.4f}, Reward Loss: {reward_loss:.4f}"
        )
        print()


def train_model(model, data, epochs=10, batch_size=32, test_size=0.01):
    X = np.array([example.board_state for example in data])
    y_move = np.array([example.move_one_hot for example in data])
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
            "move_output": "categorical_crossentropy",
            "reward_output": "mse",
        },
        loss_weights={"move_output": 0.5, "reward_output": 0.5},
        metrics={"move_output": "accuracy", "reward_output": "mse"},
    )

    test_accuracy_callback = TestAccuracyCallback(
        X_test, y_move_test, y_reward_test
    )

    model.fit(
        X_train,
        {
            "move_output": y_move_train,
            "reward_output": y_reward_train,
        },  # Changed here
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[test_accuracy_callback],
    )
    return model


def predict_next_move(model, board_state, game):
    """Predicts the next move based on the current board state using a weighted coin flip."""
    board_state_array = np.array(board_state).reshape(1, 9)
    predictions = model.predict(board_state_array, verbose=0)
    move_probabilities = predictions[0][0]  # Extract the move probabilities

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
                    if game.make_move(row, col):
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

            board_state = game.get_board_state().flatten()
            predicted_move = predict_next_move(model, board_state, game)
            row, col = predicted_move
            print(f"Model plays at: ({row}, {col})")
            assert game.make_move(row, col)

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


def inspect_data(data):
    """Inspects the generated test data and prints out a random game."""
    if not data:
        print("No data to inspect.")
        return

    random_example = random.choice(data)
    print("Random Game Inspection:")
    print("-" * 20)

    board_state = random_example.board_state.reshape((3, 3))
    print("Board State:")
    print(board_state)
    print("-" * 10)

    row, col = (
        np.where(random_example.move_one_hot == 1)[0][0] // 3,
        np.where(random_example.move_one_hot == 1)[0][0] % 3,
    )

    # Print the move and reward
    print(f"Player 2 Move: ({row}, {col})")
    print(f"Reward: {random_example.reward}")
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
