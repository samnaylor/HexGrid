import random

from math import cos, sin, sqrt, radians
from heapq import heappop, heappush
from dataclasses import dataclass

import pygame

pygame.init()
pygame.font.init()

# region Constants

hex_size = 30
hex_width = sqrt(3) * hex_size
hex_height = hex_size * 3 / 2
hex_spacing_horizontal = hex_width
hex_spacing_vertical = hex_height

window_width = 640
window_height = 480
render_offset = (-(window_width // 2), -(window_height // 2))

gamefont = pygame.font.SysFont("Helvetica", 24, False, False)

# endregion


# region utilities

def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t

# endregion


# region Coordinates

@dataclass(eq=True, order=True, unsafe_hash=True)
class Axial:
    q: float
    r: float

    @property
    def neighbours(self) -> list["Axial"]:
        return [(self + t) for t in [Axial(1, 0), Axial(1, -1), Axial(0, -1), Axial(-1, 0), Axial(-1, 1), Axial(0, 1)]]

    def dist(self, coords: "Axial") -> float:
        diff = self - coords
        return (abs(diff.q) + abs(diff.q + diff.r) + abs(diff.r)) / 2

    def lerp(self, coords: "Axial", t: float) -> "Axial":
        return Axial(lerp(self.q, coords.q, t), lerp(self.r, coords.r, t))

    def neighbour(self, i: int) -> "Axial":
        if (i < 0) or (i > 5):
            raise RuntimeError(f"neighbour index must be in range 0 - 5, not {i}")

        return self.neighbours[i]

    def round(self) -> "Axial":
        q, r, s = round(self.q), round(self.r), round(-(self.q + self.r))
        dq, dr, ds = abs(q - self.q), abs(r - self.r), abs(s - -(self.q + self.r))

        if (dq > dr) and (dq > ds):
            q = -(r + s)
        elif (dr > ds):
            r = -(q + s)

        return Axial(q, r)

    def to_pixel(self) -> tuple[float, float]:
        x = hex_size * (sqrt(3) * self.q + sqrt(3) / 2 * self.r) - render_offset[0]
        y = hex_size * (3 / 2 * self.r) - render_offset[1]

        return (x, y)

    @staticmethod
    def from_pixel(point: tuple[float, float]) -> "Axial":
        px = point[0] + render_offset[0]
        py = point[1] + render_offset[1]

        q = (sqrt(3) / 3 * px - 1 / 3 * py) / hex_size
        r = (2 / 3 * py) / hex_size

        return Axial(q, r).round()

    def __add__(self, __o: object) -> "Axial":
        match __o:
            case Axial(_q, _r):
                return Axial(self.q + _q, self.r + _r)

            case tuple(float(_q), float(_r)):
                return Axial(self.q + _q, self.r + _r)

            case _:
                raise RuntimeError()

    def __sub__(self, __o: object) -> "Axial":
        match __o:
            case Axial(_q, _r):
                return Axial(self.q - _q, self.r - _r)

            case tuple(float(_q), float(_r)):
                return Axial(self.q - _q, self.r - _r)

            case _:
                raise RuntimeError()

# endregion


# region Base Tile

class Tile:
    """
    Basic Tile class that implements only the core functionality
    """

    def __init__(self, coords: Axial) -> None:
        self.coords = coords
        self.vertices: list[tuple[float, float]] = []

        centerx, centery = self.coords.to_pixel()

        for i in range(6):
            angle = radians(60 * i - 30)
            self.vertices.append((centerx + hex_size * cos(angle), centery + hex_size * sin(angle)))

        self._cost = random.randint(1, 5)

    @property  # TODO: make abstract
    def cost(self) -> int:
        return self._cost

    def is_hovered_over(self) -> bool:
        return Axial.from_pixel(pygame.mouse.get_pos()) == self.coords

    def distance(self, tile: "Tile") -> float:
        return self.coords.dist(tile.coords)

    def render(self, surface: pygame.Surface) -> None:  # TODO: make abstract
        pygame.draw.polygon(surface, (0, 0, 255), self.vertices)
        pygame.draw.polygon(surface, (255, 255, 255), self.vertices, 2)

        text = gamefont.render(f"{self.cost}", True, (255, 255, 255))
        rect = text.get_rect(center=self.coords.to_pixel())

        surface.blit(text, rect)

# endregion


# region Tile Functions

# TODO: should be an abstract method implemented on the individual tile types?
def is_tile_reachable(tilemap: dict[Axial, Tile], origin: Tile, goal: Tile, *, movement: int = 10) -> bool:
    visited = set[Axial]()
    fringes = [[origin.coords]]

    visited.add(origin.coords)

    for k in range(1, 100):
        fringes.append([])
        for hex in fringes[k - 1]:
            for direction in range(6):
                neighbour = hex.neighbour(direction)
                exists = tilemap.get(neighbour) is not None

                if exists and (neighbour not in visited):
                    visited.add(neighbour)
                    fringes[k].append(neighbour)

    return goal.coords in visited


def draw_path(surface: pygame.Surface, path: list[Tile], colour: tuple[int, int, int]) -> None:
    points = [tile.coords.to_pixel() for tile in path]

    if len(points) >= 2:
        pygame.draw.lines(surface, colour, False, points, 3)

# endregion


# region A*

def astar_pathfinding(tilemap: dict[Axial, Tile], origin: Tile, goal: Tile, *, movement: int = 10) -> tuple[list[Tile], list[Tile]]:
    def heuristic(a: Axial, b: Axial) -> float:
        return a.dist(b)

    def get_neighbours(tile: Axial) -> list[Axial]:
        return filter(lambda t: tilemap.get(t) is not None, [tile.neighbour(i) for i in range(6)])

    origin_ = origin.coords
    goal_ = goal.coords

    frontier: list[tuple[float, Axial]] = [(0, origin_)]
    came_from = {}
    cost_so_far = {origin_: 0}

    while frontier:
        _, current = heappop(frontier)

        if current == goal_:
            break

        for neighbour in get_neighbours(current):
            if (tile := tilemap.get(neighbour)) is not None:
                new_cost = cost_so_far[current] + tile.cost

                if (neighbour not in cost_so_far) or (new_cost < cost_so_far[neighbour]):
                    cost_so_far[neighbour] = new_cost
                    priority = new_cost + heuristic(goal_, neighbour)
                    heappush(frontier, (priority, neighbour))
                    came_from[neighbour] = current

    path = [goal_]

    while (path[-1] != origin_):
        path.append(came_from[path[-1]])

    path.reverse()

    moveable = [path[0]]
    k = 1
    while cost_so_far[path[k]] <= movement:
        moveable.append(path[k])
        k += 1

        if k >= len(path):
            break

    return [tilemap[c] for c in moveable], [tilemap[c] for c in path]

# endregion


# region Unit

class Unit:
    def __init__(self, coords: Axial, movement: int) -> None:
        self.coords = coords
        self.movement = movement

        self.selected = False

    def move_to(self, tilemap: dict[Axial, Tile], goal: Tile) -> None:
        moveable, _ = astar_pathfinding(tilemap, tilemap[self.coords], goal, movement=self.movement)
        self.coords = moveable[-1].coords

    def render(self, surface: pygame.Surface) -> None:
        centerx, centery = self.coords.to_pixel()
        pygame.draw.rect(surface, (0, 255, 0), (centerx - hex_size // 2, centery - hex_size // 2, hex_size, hex_size))

        if self.selected:
            pygame.draw.rect(surface, (0, 0, 0), (centerx - hex_size // 2, centery - hex_size // 2, hex_size, hex_size), 4)

# endregion


# region entrypoint

def is_quit_event(event: pygame.event.Event) -> bool:
    return (event.type == pygame.QUIT) or ((event.type == pygame.KEYDOWN) and (event.key == pygame.K_ESCAPE))


test_tiles = list(map(lambda t: Axial(t[0], t[1]), [
    (-2, -1), (-1, -1), (0, -1), (1, -1), (2, -1), (3, -1),
    (-3, 0), (-2, 0), (-1, 0), (0, 0), (1, 0), (2, 0), (3, 0),
    (-3, 1), (-2, 1), (-1, 1), (0, 1), (1, 1), (2, 1),
]))


def generate_tilemap() -> dict[Axial, Tile]:
    return dict((coord, Tile(coord)) for coord in test_tiles)


def unit_on_tile(units: list[Unit], coord: Axial) -> bool:
    for unit in units:
        if unit.coords == coord:
            return True
    return False


def main() -> int:
    screen = pygame.display.set_mode((window_width, window_height))
    clock = pygame.time.Clock()
    running = True

    tilemap = generate_tilemap()

    units = [Unit(Axial(-2, -1), movement=5), Unit(Axial(2, 1), movement=8)]
    selected: Unit | None = None

    text = gamefont.render("Press enter to re-roll tile costs.", True, (255, 255, 255))
    rect = text.get_rect(topleft=(20, 20))

    text2 = gamefont.render("Unit Movement: N/A", True, (255, 255, 255))
    rect2 = text2.get_rect(topleft=(20, 50))

    text3 = gamefont.render("Green - 'moveable', red - 'desired'", True, (255, 255, 255))
    rect3 = text3.get_rect(topleft=(20, 80))

    while running:
        for event in pygame.event.get():
            if is_quit_event(event):
                running = False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:
                    tilemap = generate_tilemap()

            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    pos = pygame.mouse.get_pos()
                    coord = Axial.from_pixel(pos)
                    for unit in units:
                        if unit.selected:
                            if coord == unit.coords:  # deselect by clicking same tile as selected
                                selected = None
                                unit.selected = False
                            elif not unit_on_tile(units, coord):  # move to empty tile
                                unit.move_to(tilemap, tilemap[coord])
                            else:  # deselect by clicking elsewhere
                                selected = None
                                unit.selected = False
                        else:
                            if coord == unit.coords:  # select new unit - remove previous selection if applicable
                                if selected is not None:
                                    selected.selected = False
                                selected = unit
                                unit.selected = True

        screen.fill((0, 0, 0))

        for tile in tilemap.values():
            tile.render(screen)

        for unit in units:
            unit.render(screen)

        if selected is not None:
            text2 = gamefont.render(f"Unit Movement: {selected.movement}", True, (255, 255, 255))
            goal = Axial.from_pixel(pygame.mouse.get_pos())
            if (goal in tilemap.keys()) and (goal != selected.coords):
                moveable, dream = astar_pathfinding(tilemap, tilemap[selected.coords], tilemap[goal], movement=selected.movement)
                draw_path(screen, dream, (255, 0, 0))
                draw_path(screen, moveable, (0, 255, 0))

        for (t, r) in [(text, rect), (text2, rect2), (text3, rect3)]:
            screen.blit(t, r)

        pygame.display.flip()
        clock.tick(60)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# endregion
