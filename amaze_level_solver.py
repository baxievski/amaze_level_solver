import argparse
import os
import random
import xml.etree.ElementTree as ET  
from operator import itemgetter


class Node:
    def __init__(self, y, x):
        """
        """
        self.y = y
        self.x = x
        self.left = None
        self.right = None
        self.up = None
        self.down = None

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
        return f'V({self.y}, {self.x})'


class Maze:
    # TODO: split into 2 classes: Maze (only maze generation and calculation of start position), Game - (moves and everything remaining)
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
        self.reset_solved_state()
        self.nodes = {}
        self.moves_made = []
        self.done_nodes = []
        self.nodes_to_return_to = []
        self.level = 0
        self.log = False

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
        self.done_nodes = []
        y_start = self.start_position[0]
        x_start = self.start_position[1]
        if self.start_position in self.nodes:
            self.ball = self.nodes[self.start_position][0]
        else:
            self.ball = Node(y_start, x_start)
            self.nodes[self.start_position] = [self.ball, self.possible_moves(self.start_position), []]
        self.solved_state[y_start][x_start] = '9'
    
    def possible_moves(self, position):
        moves = []
        y, x = position
        for d in Maze.directions:
            y_offset = Maze.directions[d][0]
            x_offset = Maze.directions[d][1]
            if self.layout[y + y_offset][x + x_offset] in ('1', '2'):
                if d not in moves:
                    moves.append(d)
        return moves

    def move_ball(self, direction):
        """ go until the ball hits a wall """
        if direction not in self.possible_moves((self.ball.y, self.ball.x)):
            raise Exception(f'Can not move {direction}!')
        self.nodes[(self.ball.y, self.ball.x)][2].append(direction)
        if set(self.nodes[(self.ball.y, self.ball.x)][1]) == set(self.nodes[(self.ball.y, self.ball.x)][2]):
            self.done_nodes.append(self.ball)
        self.moves_made.append(direction)
        previous_ball = self.ball
        while direction in self.possible_moves((self.ball.y, self.ball.x)):
            y_offset = Maze.directions[direction][0]
            x_offset = Maze.directions[direction][1]
            y_next = self.ball.y + y_offset
            x_next = self.ball.x + x_offset
            new_ball = Node(y_next, x_next)
            self.solved_state[y_next][x_next] = '9'
            self.ball = new_ball
        if (y_next, x_next) in self.nodes:
            self.ball = self.nodes[(y_next, x_next)][0]
        else:
            self.nodes[(y_next, x_next)] = [self.ball, self.possible_moves((y_next, x_next)), []]
        setattr(previous_ball, direction, self.ball)

    def reset_moves_made(self):
        self.moves_made = []
        for pos in self.nodes:
            self.nodes[pos][2] = []
    
    def visit_all_adjacent_nodes(self, node=None):
        if node is None:
            node = self.ball
        possible_moves = self.possible_moves((node.y, node.x))
        for direction in possible_moves:
            self.ball = node
            if direction in self.nodes[(node.y, node.x)][2]:
                continue
            self.move_ball(direction)
            self.visit_all_adjacent_nodes(self.ball)

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
        self.start()
    
    def create_state_space_trees(self):
        self.create_graph()
        state_space_trees = {}
        move_trees = {}
        for pos in self.nodes.keys():
            level = 0
            state_space_tree = {}
            move_tree = {}
            state_space_tree = {level: [[self.nodes[pos][0]]]}
            move_tree = {level: [[]]}
            passed_nodes = []
            all_nodes = set([self.nodes[pos][0] for pos in self.nodes.keys()])
            while set(passed_nodes) != all_nodes:
                nodes_on_level = []
                moves_on_level = []
                for i, previous_nodes in enumerate(state_space_tree[level]):
                    node = previous_nodes[-1]
                    previous_moves = move_tree[level][i]
                    if node in passed_nodes:
                        continue
                    nodes_on_level += [previous_nodes + [getattr(node, direction), ] for direction in self.nodes[(node.y, node.x)][1]]
                    moves_on_level += [previous_moves + [direction, ] for direction in self.nodes[(node.y, node.x)][1]]
                    passed_nodes.append(node)
                level += 1
                if nodes_on_level == []:
                    break
                if level not in state_space_tree:
                    state_space_tree[level] = []
                if level not in move_tree:
                    move_tree[level] = []
                state_space_tree[level] += (nodes_on_level)
                move_tree[level] += moves_on_level
            state_space_trees[pos] = state_space_tree
            move_trees[pos] = move_tree
        self.state_space_trees = state_space_trees
        self.move_trees = move_trees
    
    def has_node_taken_all_available_directions(self, node):
        return set(self.nodes[(node.y, node.x)][1]) == set(self.nodes[(node.y, node.x)][2])

    def get_shortest_route(self, start_node, end_node):
        for level, vertices_on_level in self.state_space_trees[(start_node.y, start_node.x)].items():
            for i, vertex in enumerate(vertices_on_level):
                if end_node in vertex:
                    return (vertex, self.move_trees[(start_node.y, start_node.x)][level][i])
        return None
    
    def get_depth_for_subtree(self, node, level):
        # TODO: see about using self.level instead of level
        max_depth = {}
        n_subsequence = []
        for i, s in enumerate(self.state_space_trees[self.start_position][level]):
            if node in s:
                n_subsequence = s
                break
        for d in self.nodes[(node.y, node.x)][1]:
            cur_subseq = n_subsequence + [getattr(node, d), ]
            max_depth[d] = level
            for i, s in enumerate(self.state_space_trees[self.start_position]):
                if i <= level:
                    continue
                for j, ss in enumerate(self.state_space_trees[self.start_position][i]):
                    if ss[:len(cur_subseq)] == cur_subseq:
                        max_depth[d] = i
                        continue
        return max_depth

    def return_to_node(self, node, level):
        route_back = self.get_shortest_route(self.ball, node)
        if route_back is None:
            print(f"No route from {self.ball} to {node}")
            # FIXME: doesn't make sense
            return None
        for direction in route_back[1]:
            if self.is_solved:
                return self.walk_shallowest_subtree()
            self.move_ball(direction)
        self.level = level
        # TODO: check if there are more than 1 available direction
        self.nodes_to_return_to = self.nodes_to_return_to[:-1]
        if self.log:
            print(f"-> {self.ball} on level {self.level}")
        return self.walk_shallowest_subtree()
    
    def get_lowest_level_of_node(self, node):
        sst = self.state_space_trees[self.start_position]
        for l, nodes in sst.items():
            for n in nodes:
                if node in n:
                    return l

    def walk_shallowest_subtree(self):
        if self.log:
            print(f"is solved {self.is_solved}\n{self}")
        depth_of_current_subtree = max(self.get_depth_for_subtree(self.ball, self.level).items(), key=itemgetter(1))[1]
        if self.is_solved:
            return
        if self.level == depth_of_current_subtree - 1:
            possible_directions = self.nodes[(self.ball.y, self.ball.x)][1]
            possible_next_nodes = [getattr(self.ball, d) for d in possible_directions]
            l = self.level
            chosen_direction = None
            for d, n in zip(possible_directions, possible_next_nodes):
                if self.get_lowest_level_of_node(n) <= l+1:
                    l = self.get_lowest_level_of_node(n)
                    chosen_direction = d
            if self.log:
                print(f"Solved: {self.is_solved} - at {self.ball} on level {self.level} - taking {chosen_direction} to {getattr(self.ball, chosen_direction)}")
            self.move_ball(chosen_direction)
            self.level += 1
            return self.walk_shallowest_subtree()
        if self.level >= depth_of_current_subtree - 1:
            if self.nodes_to_return_to not in ([], None):
                node_to_return_to = self.nodes_to_return_to[-1][0]
                level_to_return_to = self.nodes_to_return_to[-1][1]
                return self.return_to_node(node_to_return_to, level_to_return_to)
            if self.log:
                print(f"{self}\nat {self.ball} on level {self.level} possible moves {self.nodes[(self.ball.y, self.ball.x)]}")
            all_nodes = set(self.nodes[n][0] for n in self.nodes)
            done_nodes = set(self.done_nodes)
            if self.ball not in done_nodes:
                if self.log:
                    print(f"Current node done: {self.ball in done_nodes}\nRemaining nodes: {all_nodes - done_nodes}")
                    # TODO: check if a move takes us straight to a node that is not in done_nodes
            return
        directions_taken = self.nodes[(self.ball.y, self.ball.x)][2]
        directions_to_nodes_that_need_returning_to = []
        for d in self.nodes[(self.ball.y, self.ball.x)][1]:
            for n in self.nodes_to_return_to:
                if n[0] == getattr(self.ball, d):
                    directions_to_nodes_that_need_returning_to.append(d)
        directions_to_done_nodes = []
        for d in self.nodes[(self.ball.y, self.ball.x)][1]:
            for n in self.done_nodes:
                if n == getattr(self.ball, d):
                    directions_to_done_nodes.append(d)
        depth_of_remaining_directions = {}
        for direction, depth in self.get_depth_for_subtree(self.ball, self.level).items():
            if direction in directions_taken:
                continue
            if direction in directions_to_done_nodes:
                continue
            if direction in directions_to_nodes_that_need_returning_to:
                continue
            depth_of_remaining_directions[direction] = depth
        chosen_direction, _ = min(depth_of_remaining_directions.items(), key=itemgetter(1))
        if len(depth_of_remaining_directions) > 1:
            self.nodes_to_return_to.append((self.ball, self.level))
        else:
            self.done_nodes.append(self.ball)
        if self.log:
            print(f"Solved: {self.is_solved} - at {self.ball} on level {self.level} - taking {chosen_direction} to {getattr(self.ball, chosen_direction)}")
        self.move_ball(chosen_direction)
        self.level += 1
        return self.walk_shallowest_subtree()


def main():
    # TODO: note limitation in README.md - valid ball positions are only vertices - otherwise the whole thing crumbles down
    parser = argparse.ArgumentParser(description='Utility for solving AMAZE levels')
    parser.add_argument("--level", help='Path to the AMAZE level xml file', required=True)
    parser.add_argument("--log", help='Enable extra printing', default=False, required=False)
    args, _ = parser.parse_known_args()
    amaze_level_path = os.path.abspath(args.level)

    if not os.path.isfile(amaze_level_path):
        raise Exception(f"File {amaze_level_path} not found")

    maze = Maze(amaze_level_path)
    if args.log:
        maze.log = True
    maze.create_state_space_trees()
    # print(maze)
    maze.walk_shallowest_subtree()
    print(f"Solved {maze.is_solved}")
    # print(f"Solved {maze.is_solved}\n{maze}")


if __name__ == '__main__':
    main()