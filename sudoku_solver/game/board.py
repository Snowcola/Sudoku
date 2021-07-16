grid = [
    [7, 8, 0, 4, 0, 0, 1, 2, 0],
    [6, 0, 0, 0, 7, 5, 0, 0, 9],
    [0, 0, 0, 6, 0, 1, 0, 7, 8],
    [0, 0, 7, 0, 4, 0, 2, 6, 0],
    [0, 0, 1, 0, 5, 0, 9, 3, 0],
    [9, 0, 4, 0, 6, 0, 0, 0, 5],
    [0, 7, 0, 3, 0, 0, 0, 1, 2],
    [1, 2, 0, 0, 0, 7, 4, 0, 0],
    [0, 4, 9, 2, 0, 6, 0, 0, 7],
]


class Board:
    def __init__(self, grid) -> None:
        self.grid = grid
        self.no_edit = {}
        for y, row in enumerate(grid):
            for x, val in enumerate(row):
                self.no_edit[(x, y)] = val == 0

    def get_row(self, row):
        return self.grid[row]

    def get_all_rows(self):
        return self.grid

    def get_col(self, col):
        return [row[col] for row in self.grid]

    def get_all_cols(self):
        return [self.get_col(x) for x in range(9)]

    def get_cell(self, pos):
        x, y = pos
        cell_x = (x // 3) * 3
        cell_y = (y // 3) * 3

        cell = []
        for row in range(cell_y, cell_y + 3):
            cell += self.get_row(row)[cell_x : cell_x + 3]
        return cell

    def get_all_cells(self):
        cells = []
        for x in range(1, 4):
            for y in range(1, 4):
                cells.append((x * 3 - 1, y * 3 - 1))

        return [self.get_cell(pos) for pos in cells]

    def is_editable(self, pos):
        return self.no_edit[pos]

    def next_empty_pos(self):
        for y in range(9):
            for x in range(9):
                if self.grid[y][x] == 0:
                    return (x, y)
        return None

    def set_value(self, pos, val, force=False):
        x, y = pos
        if self.is_editable(pos) or force:
            self.grid[y][x] = val

    def get_position(self, x, y):
        return self.grid[y][x]
