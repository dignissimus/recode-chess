import sys
import chess.pgn
from recode.analysis import analyse


def main():
    while True:
        game = chess.pgn.read_game(sys.stdin)
        if not game:
            exit()
        analysis = analyse(game)
        print(analysis)


if __name__ == "__main__":
    main()
