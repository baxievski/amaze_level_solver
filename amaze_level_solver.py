import argparse
import xml.etree.ElementTree as ET  
from copy import deepcopy
from operator import itemgetter
from time import time
from pathlib import Path


class Vertex:
    def __init__(self, y, x):
        self.y = y
        self.x = x
        self.left = None
        self.right = None
        self.up = None
        self.down = None

    def __repr__(self):
        dir_arrows = {"left": "←", "right": "→", "up": "↑", "down": "↓"}
        repr_edges = ""

        for d in dir_arrows:
            if getattr(self, d) is not None:
                repr_edges += dir_arrows[d]

        return f"V({self.y}, {self.x})"
    
    @property
    def pos(self):
        return (self.y, self.x)

    def __eq__(self, other):
        return isinstance(self, type(other)) and self.pos == other.pos

    def __hash__(self):
        return hash(self.pos)


class Maze:
    # TODO: split into: Maze (maze generation and calculation of start position), Game (moves and everything remaining)
    directions_offsets = {
        "left": (0, -1),
        "right": (0, 1),
        "up": (-1, 0),
        "down": (1, 0),
    }
    free_spaces = ["1", "2", "3"]

    def __init__(self, amaze_level_path):
        with open(amaze_level_path) as f:
            level = f.read()

        xml_etree = ET.ElementTree(ET.fromstring(level))
        etree_root = xml_etree.getroot()

        for etree_element in etree_root:
            if etree_element.tag == "layer":
                width = int(etree_element.attrib["width"])
                height = int(etree_element.attrib["height"])
                data = etree_element.find("data").text

        layout = data.replace("\n", "").split(",")
        self.layout = [layout[y * width : (y + 1) * width] for y in range(height)]
        self.reset_solved_state()
        self.vertices = {}
        self.moves_made = []
        self.explored_verticess = []
        self.vertices_to_return_to = []
        self.level = 0
        self.log = False
        self.self_before_backtrack = None

    def __repr__(self):
        rows = len(self.layout)
        columns = len(self.layout[0])
        r_layout = "\n".join(" ".join(x) for x in self.layout)
        r_layout = r_layout.replace("0", "■")

        for free_space in Maze.free_spaces:
            r_layout = r_layout.replace(free_space, " ")

        r_solved_state = deepcopy(self.solved_state)

        if self.ball is not None:
            r_solved_state[self.ball.y][self.ball.x] = "x"

        r_solved = "\n".join(" ".join(x) for x in r_solved_state)
        r_solved = r_solved.replace("1", "■")
        r_solved = r_solved.replace("9", "□")
        r_solved = r_solved.replace("x", "◍")
        r_solved = r_solved.replace("0", " ")
        return f"{r_solved}"

    def reset_solved_state(self):
        solved_state = []

        for row in self.layout:
            solved_row = []

            for element in row:
                if element == "0":
                    solved_row.append("1")
                elif element in Maze.free_spaces:
                    solved_row.append("0")
                else:
                    raise Exception(f"Illegal element: {element} in {row}.")

            solved_state.append(solved_row)

        self.solved_state = solved_state

    @property
    def start_position(self):
        for y, row in enumerate(self.layout):
            for x, element in enumerate(row):
                if element in Maze.free_spaces:
                    return (y, x)
    
    @property
    def is_solved(self):
        for row in self.solved_state:
            for el in row:
                if el == "0":
                    return False

        return True
    
    def start(self):
        if self.start_position is None:
            raise Exception("Can not place ball on level")

        self.explored_verticess = []

        y_start = self.start_position[0]
        x_start = self.start_position[1]

        if self.start_position in self.vertices:
            self.ball = self.vertices[self.start_position][0]
        else:
            self.ball = Vertex(y_start, x_start)
            possible_moves = self.possible_moves(self.start_position)
            self.vertices[self.start_position] = [self.ball, possible_moves, [], []]

        self.solved_state[y_start][x_start] = "9"
    
    def possible_moves(self, position):
        moves = []

        y, x = position

        for d in Maze.directions_offsets:
            y_offset = Maze.directions_offsets[d][0]
            x_offset = Maze.directions_offsets[d][1]
            if self.layout[y + y_offset][x + x_offset] in Maze.free_spaces:
                if d not in moves:
                    moves.append(d)

        return moves

    def move_ball(self, direction):
        """ go until the ball hits a wall """
        if direction not in self.possible_moves(self.ball.pos):
            raise Exception(f"Can not move {direction}!")

        self.vertices[self.ball.pos][2].append(direction)

        possible_moves = self.vertices[self.ball.pos][1]
        made_moves = self.vertices[self.ball.pos][2]

        if set(possible_moves) == set(made_moves):
            self.explored_verticess.append(self.ball)

        self.moves_made.append(direction)

        previous_ball = self.ball

        while direction in self.possible_moves(self.ball.pos):
            y_offset = Maze.directions_offsets[direction][0]
            x_offset = Maze.directions_offsets[direction][1]
            y_next = self.ball.y + y_offset
            x_next = self.ball.x + x_offset
            new_ball = Vertex(y_next, x_next)
            self.solved_state[y_next][x_next] = "9"
            self.ball = new_ball

        if (y_next, x_next) in self.vertices:
            self.ball = self.vertices[(y_next, x_next)][0]
        else:
            available_moves = self.possible_moves((y_next, x_next))
            self.vertices[(y_next, x_next)] = [self.ball, available_moves, [], []]

        setattr(previous_ball, direction, self.ball)

    def reset_moves_made(self):
        self.moves_made = []

        for pos in self.vertices:
            self.vertices[pos][2] = []
            self.vertices[pos][3] = []
    
    def visit_all_adjacent_vertices(self, vertex=None):
        if vertex is None:
            vertex = self.ball

        possible_moves = self.possible_moves(vertex.pos)

        for direction in possible_moves:
            self.ball = vertex

            if direction in self.vertices[vertex.pos][2]:
                continue

            self.move_ball(direction)
            self.visit_all_adjacent_vertices(self.ball)

    def create_graph(self):
        self.start()
        self.visit_all_adjacent_vertices()
        # TODO if it is not solved at this point - it will not be solvable "regularly"
        self.reset_solved_state()
        self.reset_moves_made()
        self.start()
    
    def create_spanning_trees(self):
        self.create_graph()

        spanning_trees = {}
        move_trees = {}

        for pos in self.vertices.keys():
            level = 0
            spanning_tree = {}
            move_tree = {}
            spanning_tree = {level: [[self.vertices[pos][0]]]}
            move_tree = {level: [[]]}
            visited_vertices = []
            all_vertices = set([self.vertices[pos][0] for pos in self.vertices.keys()])

            while set(visited_vertices) != all_vertices:
                vertices_on_level = []
                moves_on_level = []

                for i, prev_vertices in enumerate(spanning_tree[level]):
                    vertex = prev_vertices[-1]
                    previous_moves = move_tree[level][i]

                    if vertex in visited_vertices:
                        continue

                    vertices_on_level += [
                        prev_vertices + [getattr(vertex, direction)]
                        for direction in self.vertices[vertex.pos][1]
                    ]
                    moves_on_level += [
                        previous_moves + [direction]
                        for direction in self.vertices[vertex.pos][1]
                    ]
                    visited_vertices.append(vertex)

                level += 1

                if vertices_on_level == []:
                    break

                if level not in spanning_tree:
                    spanning_tree[level] = []

                if level not in move_tree:
                    move_tree[level] = []

                spanning_tree[level] += vertices_on_level
                move_tree[level] += moves_on_level

            spanning_trees[pos] = spanning_tree
            move_trees[pos] = move_tree

        self.spanning_trees = spanning_trees
        self.move_trees = move_trees
    
    def get_shortest_route(self, start, end):
        for level, vertices_on_level in self.spanning_trees[start.pos].items():
            for i, vertex in enumerate(vertices_on_level):
                if end in vertex:
                    moves = self.move_trees[start.pos][level][i]
                    return (vertex, moves)

        return None
    
    def subtree_depths(self, vertex):
        max_depth = {}
        n_subsequence = []

        for i, s in enumerate(self.spanning_trees[self.start_position][self.level]):
            if vertex == s[-1]:
                n_subsequence = s
                break

        for d in self.vertices[vertex.pos][1]:
            cur_subseq = n_subsequence + [getattr(vertex, d)]
            max_depth[d] = self.level

            for i, s in enumerate(self.spanning_trees[self.start_position]):
                if i <= self.level:
                    continue

                for j, ss in enumerate(self.spanning_trees[self.start_position][i]):
                    if ss[: len(cur_subseq)] == cur_subseq:
                        max_depth[d] = i
                        continue

        return max_depth

    def return_to_vertex(self, vertex, level):
        route_back = self.get_shortest_route(self.ball, vertex)

        if route_back is None:
            print(f"No route from {self.ball} to {vertex}")
            return None

        for direction in route_back[1]:
            if self.is_solved:
                return self.depth_first_walk()

            self.move_ball(direction)

        self.level = level

        v_possible_moves = self.vertices[vertex.pos][1]
        v_moves_made = self.vertices[vertex.pos][3]

        if set(v_possible_moves) == set(v_moves_made):
            self.vertices_to_return_to = self.vertices_to_return_to[:-1]

        if self.log:
            print(f"-> {self.ball} on level {self.level}")

        return self.depth_first_walk()

    def backtrack(self):
        if self.vertices_to_return_to in [None, []]:
            return self.depth_first_walk()

        vertex = self.vertices_to_return_to[-1][0]
        level = self.vertices_to_return_to[-1][1]

        v_possible_moves = self.vertices[vertex.pos][1]
        v_moves_made = self.vertices[vertex.pos][3]

        if set(v_possible_moves) == set(v_moves_made):
            self.vertices_to_return_to = self.vertices_to_return_to[:-1]
            return self.depth_first_walk()

        return self.return_to_vertex(vertex, level)

    def depth_first_walk(self):
        if self.log:
            print(f"is solved {self.is_solved}\n{self}")

        if self.is_solved:
            return

        subtree_depth = max(self.subtree_depths(self.ball).items(), key=itemgetter(1))[1]

        if self.level == subtree_depth:
            if self.vertices_to_return_to in ([], None):
                if self.log:
                    print(f"{self}\nat {self.ball} on level {self.level} possible moves {self.vertices[self.ball.pos]}")

                return

            if self.self_before_backtrack is None:
                self.self_before_backtrack = deepcopy(self)
            elif self.solved_state == self.self_before_backtrack.solved_state:
                if self.log:
                    print(f"useless backtrack\n{self}")

                self.ball = deepcopy(self.self_before_backtrack.ball)
                self.level = self.self_before_backtrack.level

                v_possible_moves = self.vertices[self.ball.pos][1]
                v_moves_made = self.vertices[self.ball.pos][3]

                if set(v_possible_moves) == set(v_moves_made):
                    self.vertices_to_return_to = self.vertices_to_return_to[:-1]

                self.moves_made = deepcopy(self.self_before_backtrack.moves_made)

                self.self_before_backtrack = None
                self.self_before_backtrack = deepcopy(self)

                return self.backtrack()

            return self.backtrack()

        directions_taken = self.vertices[self.ball.pos][3]
        depth_of_remaining_directions = {}

        for direction, depth in self.subtree_depths(self.ball).items():
            if direction in directions_taken:
                continue

            depth_of_remaining_directions[direction] = depth

        if depth_of_remaining_directions == {}:
            # End of subtreee. Vertex appeared more than once on this level. Backtrack
            if self.vertices_to_return_to in ([], None):
                if self.log:
                    print(f"{self}\nat {self.ball} on level {self.level} possible moves {self.vertices[self.ball.pos]}")

                return

            if self.self_before_backtrack is None:
                self.self_before_backtrack = deepcopy(self)
            elif self.solved_state == self.self_before_backtrack.solved_state:
                if self.log:
                    print(f"useless backtrack\n{self}")

                self.ball = deepcopy(self.self_before_backtrack.ball)
                self.level = self.self_before_backtrack.level

                v_possible_moves = self.vertices[self.ball.pos][1]
                v_moves_made = self.vertices[self.ball.pos][3]

                if set(v_possible_moves) == set(v_moves_made):
                    self.vertices_to_return_to = self.vertices_to_return_to[:-1]

                self.moves_made = deepcopy(self.self_before_backtrack.moves_made)

                self.self_before_backtrack = None
                self.self_before_backtrack = deepcopy(self)

                return self.backtrack()

            return self.backtrack()

        chosen_direction, _ = max(depth_of_remaining_directions.items(), key=itemgetter(1))

        if len(depth_of_remaining_directions) > 1:
            self.vertices_to_return_to.append((self.ball, self.level))
        else:
            self.explored_verticess.append(self.ball)

        if self.log:
            print(f"Solved: {self.is_solved} - at {self.ball} on level {self.level} - taking {chosen_direction} to {getattr(self.ball, chosen_direction)}")

        self.vertices[self.ball.pos][3].append(chosen_direction)
        self.move_ball(chosen_direction)
        self.level += 1

        return self.depth_first_walk()


def main():
    parser = argparse.ArgumentParser(description="Utility for solving AMAZE levels")
    parser.add_argument("--level", help="Path to AMAZE level xml file", required=False)
    parser.add_argument("--alllevels", help="Path to dir containing AMAZE level xml files", required=False)
    parser.add_argument("--log", help="Extra printing", default=False, required=False)

    args, _ = parser.parse_known_args()

    amaze_levels_paths = []

    if args.alllevels is not None:
        amaze_levels_paths = list(Path(args.alllevels).glob('*.xml'))
    
    if args.level is not None:
        amaze_levels_paths.append(Path(args.level))
    
    amaze_levels_paths = sorted(amaze_levels_paths)

    for amaze_level_path in amaze_levels_paths:
        time_start = time()

        if not amaze_level_path.is_file():
            raise Exception(f"File {amaze_level_path} not found")

        maze = Maze(amaze_level_path)

        if args.log:
            maze.log = True

        maze.create_spanning_trees()
        maze.depth_first_walk()

        time_end = time()

        print(f"{amaze_level_path.name} {(time_end - time_start):.1f}sec: {'solved' if maze.is_solved else 'no solution found'} in {len(maze.moves_made)} moves")


if __name__ == "__main__":
    main()
