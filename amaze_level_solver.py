import argparse
import os
import random
import xml.etree.ElementTree as ET  


class Ball:
    def __init__(self, y, x):
        """
        append to self.moves_made on each move
        """
        self.y = y
        self.x = x
        self.left = None
        self.right = None
        self.up = None
        self.down = None
        # [ ] TODO probably doesn't belong here
        self.moves_made = []

    def __repr__(self):
        dir_arrows = {
            'left': '←',
            'right': '→',
            'up': '↑',
            'down': '↓'
        }
        repr_edges = ''
        for d in ('left', 'right', 'up', 'down'):
            if getattr(self, d) is not None:
                repr_edges += dir_arrows[d]
        return f'({self.y}, {self.x}) {repr_edges}'


class Maze:
    directions = {
        'left': (0, -1),
        'right': (0, 1),
        'up': (-1, 0),
        'down': (1, 0)
    }

    def __init__(self, amaze_level_path):
        """
        """
        with open(amaze_level_path) as f:
            level = f.read()
        tree = ET.ElementTree(ET.fromstring(level))
        root = tree.getroot()
        for elem in root:
            if elem.tag == 'layer':
                width = int(elem.attrib['width'])
                height = int(elem.attrib['height'])
                data = elem.find('data').text
        _d = data.replace('\n', '').split(',')
        self.layout = [_d[y * width : (y+1) * width] for y in range(height)]
        self.ball_possitions = {}
        self.reset_solved_state()
        self.balls = {}

    def __repr__(self):
        r_layout = '\n'.join(' '.join(x) for x in self.layout)
        r_layout = r_layout.replace("0", "■")
        r_layout = r_layout.replace("1", " ")
        r_layout = r_layout.replace("2", " ")
        r_solved = '\n'.join(' '.join(x) for x in self.solved_state)
        r_solved = r_solved.replace("1", "■")
        r_solved = r_solved.replace("9", "□")
        r_solved = r_solved.replace("0", " ")
        return f"{r_solved}"

    def reset_solved_state(self):
        solved_state = []
        for row in self.layout:
            solved_row = []
            for element in row:
                if element == '0':
                    solved_row.append('1')
                elif element in ('1', '2'):
                    solved_row.append('0')
                else:
                    raise Exception(f'Illegal elemennt {element} in {row}.')
            solved_state.append(solved_row)
        self.solved_state = solved_state

    @property
    def start_position(self):
        for y, row in enumerate(self.layout):
            for x, element in enumerate(row):
                if element in ('1', '2'):
                    return (y, x)
        pass
    
    @property
    def is_solved(self):
        for row in self.solved_state:
            for el in row:
                if el == '0':
                    return False
        return True
    
    def start(self):
        if self.start_position is None:
            raise Exception("Can not place ball on level")
        y_pos = self.start_position[0]
        x_pos = self.start_position[1]
        self.ball_possitions
        self.ball = Ball(y_pos, x_pos)
        self.balls[(y_pos, x_pos)] = self.ball
        self.solved_state[y_pos][x_pos] = '9'
    
    def possible_moves(self, y, x):
        moves = {}
        for d in Maze.directions:
            y_offset = Maze.directions[d][0]
            x_offset = Maze.directions[d][1]
            if self.layout[y + y_offset][x + x_offset] in ('1', '2'):
                if d not in moves.keys():
                    moves[d] = None
        return moves

    def move_ball(self, direction):
        """ go until the ball hits a wall """
        if direction not in self.possible_moves(self.ball.y, self.ball.x):
            raise Exception(f'Can not move {direction}!')
        self.ball.moves_made.append(direction)
        previous_ball = self.ball
        while direction in self.possible_moves(self.ball.y, self.ball.x):
            y_offset = Maze.directions[direction][0]
            x_offset = Maze.directions[direction][1]
            y_next = self.ball.y + y_offset
            x_next = self.ball.x + x_offset
            new_ball = Ball(y_next, x_next)
            self.solved_state[y_next][x_next] = '9'
            self.ball = new_ball
        if (y_next, x_next) in self.balls:
            self.ball = self.balls[(y_next, x_next)]
        else:
            self.balls[(y_next, x_next)] = self.ball
        setattr(previous_ball, direction, self.ball)

    def create_graph(self):
        self.start()
        self.visit_all_adjacent_nodes()
        # print(f"----------------\n{self}\nis maze solved: {self.is_solved}\n----------------")
        # [ ] TODO think if it is possible to get solvability of maze just by looking at
        #          self.is_solved at this point
        #          maybe some node cannot be returned to after moving away from it
        #          if it is not solvable at this point - it will not be solvable "regularly"
        self.reset_solved_state()
        self.reset_moves_made()
    
    def reset_moves_made(self):
        for pos in self.balls:
            self.balls[pos].moves_made = []
    
    def visit_all_adjacent_nodes(self, node=None):
        if node is None:
            node = self.ball
        possible_moves = list(self.possible_moves(node.y, node.x).keys())
        for direction in possible_moves:
            self.ball = node
            if direction in self.ball.moves_made:
                continue
            self.move_ball(direction)
            self.visit_all_adjacent_nodes(self.ball)


def main():
    parser = argparse.ArgumentParser(description='Utility for solving AMAZE levels')
    parser.add_argument("--level", help='Path to the AMAZE level xml file', required=True)
    args, _ = parser.parse_known_args()
    amaze_level_path = os.path.abspath(args.level)

    if not os.path.isfile(amaze_level_path):
        raise Exception(f"File {amaze_level_path} not found")

    # [x] TODO solution for 043 is wrong? - probably not, chalenge requirements have start position top left, game starts bottom left - ask Eran
    # interesting levels: "165 dan.xml", "129 dan.xml", "058.xml"
    # [ ] TODO build the graph in one pass
    # [ ] TODO use an algorithm for walking the graph like A*, or dijkstra, or...
    #          cannot be pathfinding algorithm
    #          maybe something like euler's trail or chinese postman,
    #          but this problem is not the same
    #          we don't need to go over every edge in order to solve the maze...
    #          will try to convert the graph to a tree, and use a depth first search with backtracking
    # [ ] TODO convert the graph to a state space tree
    #          https://stackoverflow.com/questions/44875681/how-do-i-implement-a-state-space-tree-which-is-a-binary-tree-in-python
    maze = Maze(amaze_level_path)
    maze.start()
    maze.create_graph()
    print(maze)
    for k in maze.balls.keys():
        print(k)


if __name__ == '__main__':
    main()