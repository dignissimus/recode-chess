import math
from enum import Enum, auto

import chess
import chess.engine

LIMIT = chess.engine.Limit(time=1)


def winning_chance(evaluation):
    return (1 + (2 / (1 + math.exp(-0.00368208 * evaluation)) - 1)) / 2


def analyse(game):
    print("Scanning through game")
    engine = chess.engine.SimpleEngine.popen_uci("stockfish")
    moves = list(game.mainline_moves())
    positions = list(game.mainline())
    previous_evaluation = (
        engine.analyse(positions[0].board(), LIMIT)["score"]
        .white()
        .score(mate_score=1000)
    )
    previous_board = positions[0].board()
    for index, (move, position) in enumerate(zip(moves[1:], positions[1:])):
        board = position.board()
        info = engine.analyse(board, LIMIT)
        evaluation = info["score"].white().score(mate_score=1000)
        delta = winning_chance(evaluation) - winning_chance(previous_evaluation)
        previous_evaluation = evaluation
        if board.turn is chess.WHITE:
            # Black just played
            delta = -delta
            multiplier = -1

        advice = toadvice(delta)
        found_explanation = False
        if advice:
            print(previous_board.san(move), "was a", advice)
            for evaluator in IMMEDIATE:
                explanation = evaluator(previous_board, board)
                if explanation:
                    found_explanation = True
                    print(explanation.explain())
            for evaluator in LOOKAHEAD:
                pv = list(info["pv"])
                tactic = tactics(previous_board, board, pv)
                if tactic:
                    found_explanation = True
                    print("This move", tactic.explain())
                    print(">>> Continuation", pv)

            if not found_explanation:
                print(">> Unable to find explanation")
        else:
            print(previous_board.san(move))

        previous_board = position.board()


MIDDLE_GAME_VALUES = {
    chess.PAWN: 124,
    chess.KNIGHT: 781,
    chess.BISHOP: 925,
    chess.ROOK: 1276,
    chess.QUEEN: 2538,
    None: 0,
    chess.KING: 0,
}
REGULAR_VALUES = {
    chess.PAWN: 206,
    chess.KNIGHT: 854,
    chess.BISHOP: 915,
    chess.ROOK: 1380,
    chess.QUEEN: 2682,
    None: 0,
    chess.KING: 0,
}

BLACK_PAWN = chess.Piece(chess.PAWN, chess.BLACK)
WHITE_PAWN = chess.Piece(chess.PAWN, chess.WHITE)


def shift(square, dx, dy):
    """
    Shift a square value
    """
    dx = -dx
    dy = -dy
    return square + dx + 8 * dy


def zerosum(evaluator):
    def helper(board):
        return evaluator(board) - evaluator(colourflip(board))

    return helper


def comparison(evaluator):
    def helper(before, after):
        difference = evaluator(after) - evaluator(before)
        if before.turn is chess.BLACK:
            difference = -difference
        return difference

    return helper


def piece_at(board, square, dx=0, dy=0):
    location = shift(square, dx, dy)
    if 0 <= location <= 63:
        return board.piece_at(location)
    return None


def wrap(cls):
    def decorator(evaluator):
        def helper(*args):
            return cls(evaluator(*args))

        return helper

    return decorator


class SpaceExplainer:
    def __init__(self, difference):
        self.difference = difference

    def __bool__(self):
        # TODO: Change the threshold from 0? Maybe?
        return self.difference > 0

    def explain(self):
        # TODO: Incomplete explanation
        # note: The explanation depends on the context
        # e.g. did we lose space? gain space? remove space?
        # who lost space? how did this happen?
        return "increases space"


@wrap(SpaceExplainer)
@comparison
@zerosum
def space(board):
    # TODO
    # For space advantages, we should note
    # whether a space advantage has
    # been gained or lost and who by
    # We should note how this has happened
    # i.e. by removing squares from the opposing player
    # by using pawns to control more squares
    # or by expanding one's own space
    # Explain space advantage to the user as a term

    if total_non_pawn_material(board) < 12222:
        return 0

    piece_count = 0
    blocked_count = 0

    space_area = 0
    for square in chess.SQUARES:
        rank = chess.square_rank(square) + 1
        file = chess.square_file(square) + 1

        if (
            rank >= 2
            and rank <= 4
            and file >= 3
            and file <= 6
            and board.piece_at(square) != WHITE_PAWN
            and piece_at(board, square, -1, -1) != BLACK_PAWN
            and piece_at(board, square, 1, -1) != BLACK_PAWN
        ):
            space_area += 1
            if (
                piece_at(board, square, 0, -1) == WHITE_PAWN
                or piece_at(board, square, 0, -2) == WHITE_PAWN
                or piece_at(board, square, 0, -3) == WHITE_PAWN
            ) and not is_attacked(colourflip(board), yflip(square)):
                space_area += 1

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is not None and piece.color is chess.WHITE:
            piece_count += 1

        if board.piece_at(square) == WHITE_PAWN and (
            piece_at(board, square, 0, -1) == BLACK_PAWN
            or (
                piece_at(board, square, -1, -2) == BLACK_PAWN
                and piece_at(board, square, 1, -2) == BLACK_PAWN
            )
        ):
            blocked_count += 1
        if board.piece_at(square) == BLACK_PAWN and (
            piece_at(board, square, 0, 1) == WHITE_PAWN
            or piece_at(board, square, -1, 2) == WHITE_PAWN
            and piece_at(board, square, 1, 2) == WHITE_PAWN
        ):
            blocked_count += 1

    weight = (piece_count - 3 + min(blocked_count, 9)) ** 2 / 16
    return int(weight * space_area)


def quiescent(evaluator):
    def helper(previous_board, board, moves):
        position = board.copy()
        quiet_counter = 0
        positions_prime = [previous_board, board]
        for move in moves:
            position.push(move)
            positions_prime.append(position)
            if is_quiet(position):
                quiet_counter += 1
            else:
                quiet_counter = 0
            if quiet_counter == 6:
                break
        return evaluator(positions_prime)

    return helper


def is_quiet(position):
    move = position.pop()
    is_capture = position.is_capture(move)
    position.push(move)
    noisy = is_capture or position.is_check()
    return not noisy


def quiet_ancestor(position):
    moves = []
    while not is_quiet(position):
        moves.append(position.pop())

    ancestor = position.copy()
    while moves:
        position.push(moves.pop())

    return ancestor


PIECES = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]


def calculate_imbalance(position):
    imbalance = {piece_type: 0 for piece_type in PIECES}
    for square in chess.SQUARES:
        piece = position.piece_at(square)
        if piece and piece.piece_type in PIECES:
            if piece.color is chess.BLACK:
                imbalance[piece.piece_type] -= 1
            else:
                imbalance[piece.piece_type] += 1
    return imbalance


def imbalance_delta(first, second):
    return {
        piece_type: difference
        for piece_type in PIECES
        if (difference := second[piece_type] - first[piece_type]) != 0
    }


@quiescent
def tactics(positions):
    """
    Attempt to explain a loss in material
    """
    initial_imbalance = calculate_imbalance(quiet_ancestor(positions[0]))
    final_imbalance = calculate_imbalance(positions[-1])

    delta = imbalance_delta(initial_imbalance, final_imbalance)

    player = positions[0].turn
    multiplier = 1
    if player is chess.BLACK:
        multiplier = -1
    normalised_delta = {
        piece_type: multiplier * value for (piece_type, value) in delta.items()
    }

    # Bishops and knights are the only pieces that have the same value
    # If we win one and lose the other
    # We haven't won anything, let's consider it a trade
    if chess.BISHOP in normalised_delta and chess.KNIGHT in normalised_delta:
        if normalised_delta[chess.BISHOP] < 0:
            if normalised_delta[chess.KNIGHT] > 0:
                bishop_loss = -normalised_delta[chess.BISHOP]
                difference = min(bishop_loss, normalised_delta[chess.KNIGHT])
                normalised_delta[chess.BISHOP] += difference
                normalised_delta[chess.KNIGHT] -= difference

        if normalised_delta and normalised_delta[chess.KNIGHT] < 0:
            if normalised_delta[chess.BISHOP] > 0:
                knight_loss = -normalised_delta[chess.KNIGHT]
                difference = min(knight_loss, normalised_delta[chess.BISHOP])
                normalised_delta[chess.KNIGHT] += difference
                normalised_delta[chess.BISHOP] -= difference

    losses = []
    for (piece_type, difference) in normalised_delta.items():
        if difference >= 0:
            continue
        losses.append((chess.piece_name(piece_type), -difference))

    return TacticExplanation(losses)


class TacticExplanation:
    def __init__(self, losses):
        self.losses = losses

    def __bool__(self):
        return len(self.losses) > 0

    def explain(self):
        loss_strings = []
        for piece, amount in self.losses:
            quantifier = str(amount)
            plural = "s"
            if amount == 1:
                quantifier = "a"
                plural = ""

            loss_strings.append(f"{quantifier} {piece}{plural}")

        return "loses " + ", ".join(loss_strings)


def mobility():
    ...


IMMEDIATE = [space]
LOOKAHEAD = [tactics]
PROSPECTIVE = [mobility]
LOSSES = [mobility]


def is_attacked(board, square):
    for attacker in board.attackers(chess.WHITE, square):
        return True
    return False


def colourflip(board):
    return board.mirror()


def xflip(square):
    # TODO: Bit manipulation
    rank = chess.square_rank(square)
    file = chess.square_file(square)

    rank = 7 - rank
    return rank * 8 + file


def yflip(square):
    return chess.square_mirror(square)


def is_middle_game(board):
    # TODO: Implement
    return False


def total_non_pawn_material(board):
    values = REGULAR_VALUES
    total = 0
    if is_middle_game(board):
        values = MIDDLE_GAME_VALUES
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece and piece != chess.PAWN:
            total += values[piece.piece_type]
    return total


class Judgement(Enum):
    blunder = auto()
    mistake = auto()

    def __str__(self):
        match self:
            case Judgement.blunder:
                return "[[blunder]]"
            case Judgement.mistake:
                return "[[mistake]]"


def toadvice(delta):
    if -delta >= 0.2:
        return Judgement.blunder
    if -delta >= 0.1:
        return Judgement.mistake
