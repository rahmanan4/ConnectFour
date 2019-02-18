import numpy as np
import inspect


class AIPlayer:
    def __init__(self, player_number):
        self.player_number = player_number
        self.type = 'ai'
        self.player_string = 'Player {}:ai'.format(player_number)
        self.action_values = []

    def game_completed(self, player_num, board):
        player_win_str = '{0}{0}{0}{0}'.format(player_num)
        to_str = lambda a: ''.join(a.astype(str))

        def check_horizontal(b):
            for row in b:
                if player_win_str in to_str(row):
                    return True
            return False

        def check_verticle(b):
            return check_horizontal(b.T)

        def check_diagonal(b):
            for op in [None, np.fliplr]:
                op_board = op(b) if op else b

                root_diag = np.diagonal(op_board, offset=0).astype(np.int)
                if player_win_str in to_str(root_diag):
                    return True

                for i in range(1, b.shape[1] - 3):
                    for offset in [i, -i]:
                        diag = np.diagonal(op_board, offset=offset)
                        diag = to_str(diag.astype(np.int))
                        if player_win_str in diag:
                            return True

            return False

        return (check_horizontal(board) or
                check_verticle(board) or
                check_diagonal(board))

    def terminal_test(self, board, depth):
        if self.game_completed(1, board) or self.game_completed(2, board) or depth == 3:
            return True
        else:
            return False

    def actions(self, board):
        actions = []
        for col in range(board.shape[1]):
            if 0 in board[:, col]:
                actions.append(col)
        return actions

    def results(self, board, action):
        temp_board = np.copy(board)
        for row in reversed(range(6)):
            if temp_board[row, action] == 0:
                temp_board[row, action] = self.player_number
                break
        return temp_board

    def max_value(self, board, alpha, beta, depth):
        if self.terminal_test(board, depth):
            return self.evaluation_function(board)
        else:
            v = float('-inf')
            actions = self.actions(board)
            for a in actions:
                v = max(v, self.min_value(self.results(board, a), alpha, beta, depth + 1))
                if v >= beta:
                    return v
                alpha = max(alpha, v)
                if depth == 0:
                    self.action_values.append([a, v])
        return v

    def min_value(self, board, alpha, beta, depth):
        if self.terminal_test(board, depth):
            return self.evaluation_function(board)
        else:
            v = float('inf')
            actions = self.actions(board)
            for a in actions:
                v = min(v, self.max_value(self.results(board,a), alpha, beta, depth + 1))
                if v <= alpha:
                    return v
                beta = min(beta, v)
        return v

    def get_alpha_beta_move(self, board):
        """
        Given the current state of the board, return the next move based on
        the alpha-beta pruning algorithm

        This will play against either itself or a human player

        INPUTS:
        board - a numpy array containing the state of the board using the
                following encoding:
                - the board maintains its same two dimensions
                    - row 0 is the top of the board and so is
                      the last row filled
                - spaces that are unoccupied are marked as 0
                - spaces that are occupied by player 1 have a 1 in them
                - spaces that are occupied by player 2 have a 2 in them

        RETURNS:
        The 0 based index of the column that represents the next move
        """

        self.actions_values = []
        v = self.max_value(board, float('-inf'), float('inf'), 0)
        for action_value in self.action_values:
            if action_value[1] == v:
                return action_value[0]

    def get_expectimax_move(self, board):
        """
        Given the current state of the board, return the next move based on
        the expectimax algorithm.

        This will play against the random player, who chooses any valid move
        with equal probability

        INPUTS:
        board - a numpy array containing the state of the board using the
                following encoding:
                - the board maintains its same two dimensions
                    - row 0 is the top of the board and so is
                      the last row filled
                - spaces that are unoccupied are marked as 0
                - spaces that are occupied by player 1 have a 1 in them
                - spaces that are occupied by player 2 have a 2 in them

        RETURNS:
        The 0 based index of the column that represents the next move
        """

        self.actions_values = []
        v = self.value(board, 0, 'max')
        for action_value in self.action_values:
            if action_value[1] == v:
                return action_value[0]

    def value(self, board, depth, agent):
        if self.terminal_test(board, depth):
            return self.evaluation_function(board)
        elif agent == 'max':
            return self.expmax_value(board, depth)
        elif agent == 'exp':
            return self.exp_value(board, depth)

    def expmax_value(self, board, depth):
        v = float('-inf')
        actions = self.actions(board)
        for a in actions:
            v = max(v, self.value(self.results(board, a), depth + 1, 'exp'))
            if depth == 0:
                self.action_values.append([a, v])
        return v

    def exp_value(self, board, depth):
        v = 0
        for action in self.actions(board):
            p = self.probability(board)
            v = v + (p*self.value(self.results(board, action), depth + 1, 'max'))
        return v

    def probability(self, board):
        p = 1/len(self.actions(board))
        return p

    def evaluation_function(self, board):
        """
        Given the current state of the board, return the scalar value that
        represents the evaluation function for the current player

        INPUTS:
        board - a numpy array containing the state of the board using the
                following encoding:
                - the board maintains its same two dimensions
                    - row 0 is the top of the board and so is
                      the last row filled
                - spaces that are unoccupied are marked as 0
                - spaces that are occupied by player 1 have a 1 in them
                - spaces that are occupied by player 2 have a 2 in them

        RETURNS:
        The utility value for the current board
        """
        w0 = 18
        w1 = 12
        w2 = 6
        w3 = 24
        w4 = 15
        w5 = 6
        w6 = 6
        w7 = 6

        f0 = 0
        f1 = 0
        f2 = 0
        f3 = 0
        f4 = 0
        f5 = 0
        f6 = 0
        f7 = 0

        def other_player(player_num):
            if player_num == 1:
                return 2
            elif player_num == 2:
                return 1

        if self.game_completed(other_player(self.player_number), board):
            f0 = 1
        if self.check_three(other_player(self.player_number), board):
            f1 = 1
        if self.check_two(other_player(self.player_number), board):
            f2 = 1

        if self.game_completed(self.player_number, board):
            f3 = 1
        if self.check_three(self.player_number, board):
            f4 = 1
        if self.check_two(self.player_number, board):
            f5 = 1

        if self.player_number in board[:, 0]:
            f6 = -1

        if self.player_number in board[:, 6]:
            f7 = -1

        utility_value = ((w0*f0) + (w1*f1) + (w2*f2) + (w3*f3) + (w4*f4) + (w5*f5) + (w6*f6) + (w7*f7) + (w10*f10))
        return utility_value

    def check_three(self, player_num, board):
        player_win_str = '{0}{0}{0}'.format(player_num)
        to_str = lambda a: ''.join(a.astype(str))

        def check_horizontal(b):
            for row in b:
                if player_win_str in to_str(row):
                    return True
            return False

        def check_verticle(b):
            return check_horizontal(b.T)

        def check_diagonal(b):
            for op in [None, np.fliplr]:
                op_board = op(b) if op else b

                root_diag = np.diagonal(op_board, offset=0).astype(np.int)
                if player_win_str in to_str(root_diag):
                    return True

                for i in range(1, b.shape[1] - 3):
                    for offset in [i, -i]:
                        diag = np.diagonal(op_board, offset=offset)
                        diag = to_str(diag.astype(np.int))
                        if player_win_str in diag:
                            return True

            return False

        return (check_horizontal(board) or
                check_verticle(board) or
                check_diagonal(board))

    def check_two(self, player_num, board):
        player_win_str = '{0}{0}'.format(player_num)
        to_str = lambda a: ''.join(a.astype(str))

        def check_horizontal(b):
            for row in b:
                if player_win_str in to_str(row):
                    return True
            return False

        def check_verticle(b):
            return check_horizontal(b.T)

        def check_diagonal(b):
            for op in [None, np.fliplr]:
                op_board = op(b) if op else b

                root_diag = np.diagonal(op_board, offset=0).astype(np.int)
                if player_win_str in to_str(root_diag):
                    return True

                for i in range(1, b.shape[1] - 3):
                    for offset in [i, -i]:
                        diag = np.diagonal(op_board, offset=offset)
                        diag = to_str(diag.astype(np.int))
                        if player_win_str in diag:
                            return True

            return False

        return (check_horizontal(board) or
                check_verticle(board) or
                check_diagonal(board))

    def check_one(self, player_num, board):
        player_win_str = '{0}'.format(player_num)
        to_str = lambda a: ''.join(a.astype(str))

        def check_horizontal(b):
            for row in b:
                if player_win_str in to_str(row):
                    return True
            return False

        def check_verticle(b):
            return check_horizontal(b.T)

        def check_diagonal(b):
            for op in [None, np.fliplr]:
                op_board = op(b) if op else b

                root_diag = np.diagonal(op_board, offset=0).astype(np.int)
                if player_win_str in to_str(root_diag):
                    return True

                for i in range(1, b.shape[1] - 3):
                    for offset in [i, -i]:
                        diag = np.diagonal(op_board, offset=offset)
                        diag = to_str(diag.astype(np.int))
                        if player_win_str in diag:
                            return True

            return False

        return (check_horizontal(board) or
                check_verticle(board) or
                check_diagonal(board))


class RandomPlayer:
    def __init__(self, player_number):
        self.player_number = player_number
        self.type = 'random'
        self.player_string = 'Player {}:random'.format(player_number)

    def get_move(self, board):
        """
        Given the current board state select a random column from the available
        valid moves.

        INPUTS:
        board - a numpy array containing the state of the board using the
                following encoding:
                - the board maintains its same two dimensions
                    - row 0 is the top of the board and so is
                      the last row filled
                - spaces that are unoccupied are marked as 0
                - spaces that are occupied by player 1 have a 1 in them
                - spaces that are occupied by player 2 have a 2 in them

        RETURNS:
        The 0 based index of the column that represents the next move
        """
        valid_cols = []
        for col in range(board.shape[1]):
            if 0 in board[:, col]:
                valid_cols.append(col)

        return np.random.choice(valid_cols)


class HumanPlayer:
    def __init__(self, player_number):
        self.player_number = player_number
        self.type = 'human'
        self.player_string = 'Player {}:human'.format(player_number)

    def get_move(self, board):
        """
        Given the current board state returns the human input for next move

        INPUTS:
        board - a numpy array containing the state of the board using the
                following encoding:
                - the board maintains its same two dimensions
                    - row 0 is the top of the board and so is
                      the last row filled
                - spaces that are unoccupied are marked as 0
                - spaces that are occupied by player 1 have a 1 in them
                - spaces that are occupied by player 2 have a 2 in them

        RETURNS:
        The 0 based index of the column that represents the next move
        """

        valid_cols = []
        for i, col in enumerate(board.T):
            if 0 in col:
                valid_cols.append(i)

        move = int(input('Enter your move: '))

        while move not in valid_cols:
            print('Column full, choose from:{}'.format(valid_cols))
            move = int(input('Enter your move: '))

        return move