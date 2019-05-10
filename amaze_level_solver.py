import random
import os
import xml.etree.ElementTree as ET  
from collections import defaultdict


class Ball:
    def __init__(self, y, x, previous_ball=None, next_ball=None):
        self.y = y
        self.x = x
        self.previous_ball = previous_ball
        self.next_ball = next_ball

    def __repr__(self):
        return f'({self.y}, {self.x})'


class Maze:
    directions = {
        'left': (0, -1),
        'right': (0, 1),
        'up': (-1, 0),
        'down': (1, 0)
    }
    oposite_directions = {
        'left': 'right',
        'right': 'left',
        'up': 'down',
        'down': 'up'
    }

    def __init__(self, data, width, height):
        _d = data.replace('\n', '').split(',')
        self.layout = [_d[y * width : (y+1) * width] for y in range(height)]
        _solved_state = []
        for row in self.layout:
            _solved_row = []
            for el in row:
                if el == '0':
                    _solved_row.append('1')
                elif el in ('1', '2'):
                    _solved_row.append('0')
            _solved_state.append(_solved_row)
        self.solved_state = _solved_state

    def __repr__(self):
        _r_layout = '\n'.join(' '.join(x) for x in self.layout)
        _r_solved = '\n'.join(' '.join(x) for x in self.solved_state)
        return _r_layout + '\n\n' + _r_solved
        
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
        self.ball = Ball(y_pos, x_pos)
        self.solved_state[y_pos][x_pos] = '1'
    
    def possible_moves(self, y, x):
        _r = []
        for d in Maze.directions:
            y_offset = Maze.directions[d][0]
            x_offset = Maze.directions[d][1]
            if self.layout[y + y_offset][x + x_offset] in ('1', '2'):
                _r.append(d)
        return _r

    def move_ball(self, direction):
        if direction not in self.possible_moves(self.ball.y, self.ball.x):
            raise Exception(f'Can not move in {direction} direction')
        while direction in self.possible_moves(self.ball.y, self.ball.x):
            y_offset = Maze.directions[direction][0]
            x_offset = Maze.directions[direction][1]
            y_next = self.ball.y + y_offset
            x_next = self.ball.x + x_offset
            new_ball = Ball(y_next, x_next, previous_ball=self.ball)
            self.ball.next_ball = new_ball
            self.ball = new_ball
            self.solved_state[y_next][x_next] = '1'


def main():
#     width = 9
#     height = 9
#     data = """
# 0,0,0,0,0,0,0,0,0,
# 0,1,1,1,0,1,1,1,0,
# 0,1,1,1,1,1,1,1,0,
# 0,1,0,0,0,1,1,1,0,
# 0,1,1,1,1,1,1,0,0,
# 0,0,0,0,0,0,1,1,0,
# 0,1,1,1,0,0,1,1,0,
# 0,1,0,1,1,1,1,1,0,
# 0,0,0,0,0,0,0,0,0
# """
    # [ ] TODO maybe move this in class init
    script_base_dir = os.path.dirname(__file__)
    level_file_path = os.path.join(script_base_dir, "amaze_levels", "142.xml")
    with open(level_file_path) as f:
        level = f.read()
    tree = ET.ElementTree(ET.fromstring(level))
    root = tree.getroot()
    for elem in root:
        if elem.tag == 'layer':
            width = int(elem.attrib['width'])
            height = int(elem.attrib['height'])
            data = elem.find('data').text
    # [ ] TODO - not good, try to build the graph in one pass, and then use an algorithm for walking the graph like A*, or dijkstra, or...
    solution = None
    prob_coeficient = 0.001
    for i in range(10_000):
        maze = Maze(data, width, height)
        maze.start()
        # print(f'{i} - ball position: {maze.ball} \n {maze}')
        current_solution = []
        prob_weights = {}
        for n in range(101):
            possible_moves = maze.possible_moves(maze.ball.y, maze.ball.x)
            if (maze.ball.y, maze.ball.x) not in prob_weights:
                prob_weights[(maze.ball.y, maze.ball.x)] = [1 for _ in possible_moves]
            move = random.choices(population=possible_moves, weights=prob_weights[(maze.ball.y, maze.ball.x)])[0]
            prob_weights[(maze.ball.y, maze.ball.x)][possible_moves.index(move)] = prob_weights[(maze.ball.y, maze.ball.x)][possible_moves.index(move)] * prob_coeficient
            # print(f' n: {n:<4} {str(possible_moves):<20} {move:<10} {str(maze.ball):<10} {str(prob_weights[(maze.ball.y, maze.ball.x)]):<10} {str(maze.is_solved):<10}')
            maze.move_ball(move)
            possible_moves = maze.possible_moves(maze.ball.y, maze.ball.x)
            if (maze.ball.y, maze.ball.x) not in prob_weights:
                prob_weights[(maze.ball.y, maze.ball.x)] = [1 for _ in possible_moves]
            prob_weights[(maze.ball.y, maze.ball.x)][possible_moves.index(maze.oposite_directions[move])] = prob_weights[(maze.ball.y, maze.ball.x)][possible_moves.index(maze.oposite_directions[move])] * prob_coeficient
            current_solution.append(move)
            if maze.is_solved:
                print(i, len(current_solution))
                if solution is None:
                    solution = current_solution
                elif len(current_solution) < len(solution):
                    solution = current_solution
                break
        print(i, len(current_solution))
        # print(f'new maze state:\n{maze}')
    print(f'solution: {solution}')


if __name__ == '__main__':
    main()