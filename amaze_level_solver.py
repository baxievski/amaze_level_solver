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


def main():
    # TODO: note limitation in README.md - valid ball positions are only vertices - otherwise the whole thing crumbles down
    parser = argparse.ArgumentParser(description='Utility for solving AMAZE levels')
    parser.add_argument("--level", help='Path to the AMAZE level xml file', required=True)
    args, _ = parser.parse_known_args()
    amaze_level_path = os.path.abspath(args.level)

    if not os.path.isfile(amaze_level_path):
        raise Exception(f"File {amaze_level_path} not found")

    maze = Maze(amaze_level_path)
    maze.create_state_space_trees()
    level = 0
    state_space_tree = maze.state_space_trees[maze.start_position]
    visited_nodes = [maze.ball, ]
    finished_nodes = []
    need_returning_nodes = []
    while not maze.is_solved:
        print(f"-------{maze.ball}, {maze.has_node_taken_all_available_directions(maze.ball)}\n{maze}")
        not_needed_directions = []
        current_position = (maze.ball.y, maze.ball.x)
        nodes_on_level = [n[-1] for n in state_space_tree[level]]
        possible_directions = maze.nodes[(maze.ball.y, maze.ball.x)][1]
        taken_directions = maze.nodes[(maze.ball.y, maze.ball.x)][2]
        subtrees_depths = maze.get_depth_for_subtree(maze.ball, level)
        for taken_direction in taken_directions:
            if taken_direction in subtrees_depths:
                del subtrees_depths[taken_direction]
        for d in possible_directions:
            if getattr(maze.ball, d) in finished_nodes + [x[0] for x in need_returning_nodes]:
                not_needed_directions.append(d)
                if d in subtrees_depths and len(subtrees_depths) > 1:
                    del subtrees_depths[d]
        # previous_node = maze.ball
        if not subtrees_depths:
            path_back = maze.get_shortest_route(maze.ball, need_returning_nodes[0][0])
            # print(f"Return to {need_returning_nodes[0]}: {path_back}")
            level = need_returning_nodes[0][1]
            for d in path_back[1]:
                previous_node = maze.ball
                maze.move_ball(d)
                print(f"{d}, {maze.ball}, {maze.has_node_taken_all_available_directions(maze.ball)}\n{maze}")
                visited_nodes.append(maze.ball)
                returned_nodes = []
                for n in need_returning_nodes:
                    if maze.has_node_taken_all_available_directions(previous_node):
                        returned_nodes.append(n)
                    if n[0] == maze.ball:
                        returned_nodes.append(n)
                for n in returned_nodes:
                    need_returning_nodes.remove(n)
            continue
        chosen_direction, _ = min(subtrees_depths.items(), key=itemgetter(1))
        if len(subtrees_depths) == 1:
            previous_node = maze.ball
            maze.move_ball(chosen_direction)
            print(f"{chosen_direction}, {maze.ball}, {maze.has_node_taken_all_available_directions(maze.ball)}\n{maze}")
            visited_nodes.append(maze.ball)
            finished_nodes.append(previous_node)
            for n in need_returning_nodes:
                if n[0] == previous_node:
                    need_returning_nodes.remove(n)
            if level == len(state_space_tree) - 1:
                # print("need to return from here - reached the bottom of the tree")
                if need_returning_nodes == []:
                    continue
                path_back = maze.get_shortest_route(maze.ball, need_returning_nodes[0][0])
                # print(f"Return to {need_returning_nodes[0]}: {path_back}")
                level = need_returning_nodes[0][1]
                for d in path_back[1]:
                    maze.move_ball(d)
                    print(f"{d}, {maze.ball}, {maze.has_node_taken_all_available_directions(maze.ball)}\n{maze}")
                    visited_nodes.append(maze.ball)
                    returned_nodes = []
                    for n in need_returning_nodes:
                        if n[0] == maze.ball:
                            returned_nodes.append(n)
                    for n in returned_nodes:
                        need_returning_nodes.remove(n)
                continue
            level += 1
            continue
        for d in subtrees_depths:
            if d != chosen_direction:
                other_direction = d
                break
        # TODO: check if it can be returned before adding it to the stack, if not - choose the other path
        route = maze.get_shortest_route(getattr(maze.ball, chosen_direction), getattr(maze.ball, other_direction))
        need_returning_nodes.append((getattr(maze.ball, other_direction), level + 1))
        maze.move_ball(chosen_direction)
        print(f"{chosen_direction}, {maze.ball}, {maze.has_node_taken_all_available_directions(maze.ball)}\n{maze}")
        visited_nodes.append(maze.ball)
        continue
        if set(maze.nodes[current_position][1]) == set(maze.nodes[current_position][2] + not_needed_directions):
            finished_nodes.append(previous_node)
            level += 1
            continue
        break

    print(f"{maze.is_solved}")
    print(maze.moves_made)
    print(maze)


if __name__ == '__main__':
    main()