import pygame
from board import Board

pygame.font.init()

screen = pygame.display.set_mode((500, 500))

pygame.display.set_caption("SUDOKU")
# icon = pygame.image.load('icon.png')
# pygame.display.set_icon(icon)
font1 = pygame.font.SysFont("comicsans", 40)


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


class Game:
    x, y = 0, 0
    dif = 500 / 9
    _val = 0

    def __init__(self, grid) -> None:
        self.grid = grid
        self.board = Board(grid)
        self.running = True
        self.win = False

    def place_guess(self, val, pos):
        self.board.set_value(pos, val)

    def clear_cell(self):
        pos = (self.x, self.y)
        self.place_guess(0, pos)

    def draw_selection_box(self):
        x = self.x
        y = self.y
        dif = self.dif

        if self.board.is_editable((x, y)):
            color = (0, 255, 0)
        else:
            color = (255, 0, 0)

        for i in range(2):
            pygame.draw.line(
                screen,
                color,
                (x * dif - 3, (y + i) * dif),
                (x * dif + dif + 3, (y + i) * dif),
                7,
            )
            pygame.draw.line(
                screen,
                color,
                ((x + i) * dif, y * dif),
                ((x + i) * dif, y * dif + dif),
                7,
            )

    def draw_value(self, pos):
        x, y = pos
        val = self.board.get_position(*pos)
        contents = font1.render(str(grid[y][x]), 1, (0, 0, 0))
        screen.blit(contents, (x * self.dif + 21, y * self.dif + 15))

    def draw_board(self):
        def bg_color(pos):
            if self.board.is_editable(pos):
                return (255, 255, 255)
            return (220, 220, 220)

        for row in range(len(self.board.grid)):
            for col in range(len(self.board.grid[row])):
                if self.board.get_position(col, row) != 0:
                    pygame.draw.rect(
                        screen,
                        bg_color((col, row)),
                        (col * self.dif, row * self.dif, self.dif + 1, self.dif + 1),
                    )
                    self.draw_value((col, row))

        for row in range(10):
            if row % 3 == 0:
                thick = 7
            else:
                thick = 1

            pygame.draw.line(
                screen, (0, 0, 0), (0, row * self.dif), (500, row * self.dif), thick
            )
            pygame.draw.line(
                screen, (0, 0, 0), (row * self.dif, 0), (row * self.dif, 500), thick
            )

    def select_xy(self, pos):
        self.x = int(pos[0] // self.dif)
        self.y = int(pos[1] // self.dif)

    def is_editable(self, pos=None):
        if not pos:
            pos = (self.x, self.y)
        return self.no_edit[pos]

    def handle_input(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            if event.type == pygame.MOUSEBUTTONDOWN:
                pos = pygame.mouse.get_pos()
                self.select_xy(pos)
                print(self.board.get_cell((self.x, self.y)))
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False

                if event.key == pygame.K_LEFT:
                    if self.x > 0:
                        self.x -= 1

                if event.key == pygame.K_RIGHT:
                    if self.x < 8:
                        self.x += 1

                if event.key == pygame.K_UP:
                    if self.y > 0:
                        self.y -= 1

                if event.key == pygame.K_DOWN:
                    if self.y < 8:
                        self.y += 1

                if event.key == pygame.K_1:
                    self._val = 1
                    self.place_guess(1, (self.x, self.y))
                if event.key == pygame.K_2:
                    self.place_guess(2, (self.x, self.y))
                if event.key == pygame.K_3:
                    self.place_guess(3, (self.x, self.y))
                if event.key == pygame.K_4:
                    self.place_guess(4, (self.x, self.y))
                if event.key == pygame.K_5:
                    self.place_guess(5, (self.x, self.y))
                if event.key == pygame.K_6:
                    self.place_guess(6, (self.x, self.y))
                if event.key == pygame.K_7:
                    self.place_guess(7, (self.x, self.y))
                if event.key == pygame.K_8:
                    self.place_guess(8, (self.x, self.y))
                if event.key == pygame.K_9:
                    self.place_guess(9, (self.x, self.y))
                if event.key in [
                    pygame.K_0,
                    pygame.K_KP_0,
                    pygame.K_DELETE,
                    pygame.K_BACKSPACE,
                ]:
                    self.clear_cell()
                if event.key == pygame.K_RETURN:
                    self.flag2 = 1
                # If R pressed clear the sudoku board
                if event.key == pygame.K_r:
                    self.rs = 0
                    self.error = 0
                    self.flag2 = 0
                    self.grid = [
                        [0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0],
                    ]
                # If D is pressed reset the board to default
                if event.key == pygame.K_d:
                    self.rs = 0
                    self.error = 0
                    self.flag2 = 0
                    self.grid = [
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

    def possible(self, pos, val):
        x, y = pos
        row = val in self.board.get_row(y)
        col = val in self.board.get_col(x)
        cell = val in self.board.get_cell(pos)
        result = not any([col, row, cell])
        return result

    def draw_all(self):
        screen.fill((255, 255, 255))
        self.draw_board()
        self.draw_selection_box()
        pygame.display.update()

    def run(self):
        self.running = True

        while self.running:
            self.handle_input()
            self.draw_all()
            self.win = self.check_win()
        pygame.quit()

    def solve_game(self):
        solved = False
        while self.running:
            self.handle_input()
            if not solved:
                solved = self.solve()

    def check_win(self):
        rows = [set(row) for row in self.board.get_all_rows() if len(set(row)) == 9]
        cols = [set(col) for col in self.board.get_all_cols() if len(set(col)) == 9]
        cells = [
            set(cell) for cell in self.board.get_all_cells() if len(set(cell)) == 9
        ]
        return all([len(rows) == 9, len(cols) == 9, len(cells) == 9])

    def solve(self):
        next_empty = self.board.next_empty_pos()
        if not next_empty:
            return True

        pygame.event.pump()
        for n in range(1, 10):
            if self.possible(next_empty, n):
                self.board.set_value(next_empty, n)
                self.x, self.y = next_empty
                self.draw_all()
                pygame.time.delay(100)
                if self.solve():
                    return True
                self.board.set_value(next_empty, 0)
                self.draw_all()
                pygame.time.delay(70)
        return False


if __name__ == "__main__":
    game = Game(grid)
    # game.run()
    game.solve_game()
