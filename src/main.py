from enum import IntEnum
from math import cos, sin, dist, sqrt, radians
from typing import Final, Optional
from dataclasses import dataclass

import pygame

pygame.init()
pygame.font.init()


WINDOW_W = 640
WINDOW_H = 480
FPS = 60

GAME_FONT = pygame.sysfont.SysFont("Cascadia Code", 24, False, False)

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREY = (104, 104, 104)
DARKER_GREY = (34, 34, 34)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
DULL_RED = (73, 25, 25)
DULL_BLUE = (25, 25, 73)
DULL_GREEN = (25, 73, 25)

NEIGHBOURS = [
    (0, -1),  # top left
    (-1, 0),  # left  
    (-1, 1),  # bottom left
    (0, 1),   # bottom right
    (1, 0),   # right
    (1, -1)   # top right
]


def naive_colour_contrast(rgb: tuple[int, int, int]) -> tuple[int, int, int]:
    r, g, b = rgb
    return (255 - r, 255 - g, 255 - b)


class TileType(IntEnum):
    WALKABLE = 0
    NON_WALKABLE = 1


@dataclass
class Point:
    x: float
    y: float

    def unpack(self) -> tuple[float, float]:
        return (self.x, self.y)

    def __add__(self, __o: object) -> "Point":
        match __o:
            case Point(_x, _y) | tuple((_x, _y)) | list([_x, _y]):
                return Point(self.x + _x, self.y + _y)

            case _:
                raise ArithmeticError("...")


@dataclass
class AxialCoordinates:
    q: int
    r: int

    def dist(self, other: "AxialCoordinates") -> float:
        vec = self - other
        return (abs(vec.q) + abs(vec.q + vec.r) + abs(vec.r)) / 2

    def neighbour(self, direction: tuple[int, int] | list[int]) -> "AxialCoordinates":
        return self + direction

    def __add__(self, __o: object) -> "AxialCoordinates":
        match __o:
            case AxialCoordinates(_q, _r) | tuple((_q, _r)) | list([_q, _r]):
                return AxialCoordinates(self.q + _q, self.r + _r)

            case _:
                raise ArithmeticError("...")

    def __sub__(self, __o: object) -> "AxialCoordinates":
        match __o:
            case AxialCoordinates(_q, _r) | tuple((_q, _r)) | list([_q, _r]):
                return AxialCoordinates(self.q - _r, self.r - _r)

            case _:
                raise ArithmeticError("...")

class Hex:
    radius: Final = 30.0
    width: Final = sqrt(3) * radius
    height: Final = 2.0 * radius

    def __init__(self, coords: AxialCoordinates, *, tile_type: TileType = TileType.WALKABLE) -> None:
        self.coords = coords
        self.tile_type = tile_type

        self.center = self.to_pixel()

        self.render_colour = WHITE
        self.border_colour = BLACK

    def set_render_colour(self, colour: tuple[int, int, int]) -> None:
        self.render_colour = colour
        self.border_colour = naive_colour_contrast(self.render_colour)

    def corner(self, i: int) -> Point:
        angle = radians(60 * i - 30)
        return Point(self.center.x + self.radius * cos(angle), self.center.y + self.radius * sin(angle))

    def to_pixel(self) -> Point:
        x = self.radius * ((sqrt(3) * self.coords.q) + (sqrt(3) * self.coords.r / 2))
        y = self.radius * (3 / 2 * self.coords.r)

        return Point(x, y)

    def hover(self, *, offset: tuple[int, int] = (0, 0)) -> bool:
        return dist((self.center + offset).unpack(), pygame.mouse.get_pos()) <= (self.radius * cos(radians(30)))

    def render(self, surface: pygame.Surface, *, offset: tuple[int, int] = (0, 0)) -> None:
        verts = [(self.corner(i) + offset).unpack() for i in range(6)]
        pygame.draw.polygon(surface, self.render_colour, verts)
        pygame.draw.polygon(surface, self.border_colour, verts, 3)  # NOTE: Borders don't always get to fully render

        if self.tile_type == TileType.WALKABLE:
            text = GAME_FONT.render(f"{self.coords.q}, {self.coords.r}", True, self.border_colour)
            rect = text.get_rect()
            rect.center = (self.center + offset).unpack()
            surface.blit(text, rect)

    def __repr__(self) -> str:
        return f"Hex(q={self.coords.q}, r={self.coords.r})"


def get_neighbours(hexmap: dict[tuple[int, int], Hex], hex: Hex) -> list[Hex]:
    neighbours: list[Hex] = []
    
    for direction in NEIGHBOURS:
        coordinate = hex.coords.neighbour(direction)
        neighbour = hexmap.get((coordinate.q, coordinate.r))

        if neighbour is not None:
            neighbours.append(neighbour)

    return neighbours


class Node:
    def __init__(self, current: Hex, origin: Hex, destination: Hex, path_cost: int) -> None:
        self.current = current
        self.origin = origin
        self.destination = destination
        self.path_cost = path_cost
        
        self.parent: Node | None = None

        self.base_cost = 1 if current.tile_type == TileType.WALKABLE else 9999
        self.cost_from_origin = int(current.coords.dist(origin.coords))
        self.cost_to_destination = int(current.coords.dist(destination.coords))

    def get_cost(self) -> int:
        return self.path_cost + self.base_cost + self.cost_from_origin + self.cost_to_destination


class Pathfinder:
    def __init__(self, hexmap: dict[tuple[int, int], Hex]) -> None:
        self.hexmap = hexmap

    def find_path(self, origin: Hex, destination: Hex) -> list[Hex]:
        nodes_not_evaluated: dict[Hex, Node] = {}
        nodes_evaluated: dict[Hex, Node] = {}

        start_node = Node(origin, origin, destination, 0)
        nodes_not_evaluated[origin] = start_node

        path: list[Hex] = []
        
        while not self.evaluate_next_node(nodes_not_evaluated, nodes_evaluated, origin, destination, path):
            continue

        return path

    def evaluate_next_node(
        self,
        nodes_not_evaluated: dict[Hex, Node],
        nodes_evaluated: dict[Hex, Node],
        origin: Hex,
        destination: Hex,
        path: list[Hex]
    ) -> bool:
        current_node = self.get_cheapest_node(nodes_not_evaluated)

        if current_node is None:
            path.clear()
            return False

        del nodes_not_evaluated[current_node.current]
        nodes_evaluated[current_node.current] = current_node

        path.clear()

        assert current_node is not None

        if current_node.current == destination:
            path.append(current_node.current)
            while (current_node is not None) and (current_node.current != origin):
                assert current_node.parent is not None
                path.append(current_node.parent.current)
                current_node = current_node.parent

            return True

        neighbours: list[Node] = []
        for tile in get_neighbours(self.hexmap, current_node.current):
            neighbours.append(Node(tile, origin, destination, current_node.get_cost()))

        for node in neighbours:
            if node in nodes_evaluated:
                continue

            if (node.get_cost() < current_node.get_cost()) or (not (node.current in nodes_not_evaluated)):
                node.parent = current_node
                if not (node.current in nodes_not_evaluated):
                    nodes_not_evaluated[node.current] = node

        return False

    def get_cheapest_node(self, nodes_not_evaluated: dict[Hex, Node]) -> Node | None:
        if not nodes_not_evaluated:
            return None

        nodes = list(nodes_not_evaluated.values())
        selected = nodes[0]

        for i in nodes[1:]:
            if i.get_cost() < selected.get_cost():
                selected = i
            elif (i.get_cost() == selected.get_cost()) and (i.cost_to_destination < selected.cost_to_destination):
                selected = i

        return selected


def a_star(hexmap: dict[tuple[int, int], Hex], origin: Hex | None, destination: Hex | None) -> list[Hex]:
    if origin is None and destination is None:
        return []

    if origin is not None:
        origin.set_render_colour(DULL_RED)

    if destination is not None:
        destination.set_render_colour(DULL_GREEN)

    path: list[Hex] = []

    if origin is not None and destination is not None:
        path = Pathfinder(hexmap).find_path(origin, destination)

    return path


def _is_quit_event(event: pygame.event.Event) -> bool:
    return (event.type == pygame.QUIT) or ((event.type == pygame.KEYDOWN) and (event.key == pygame.K_ESCAPE))


def generate_hex_map(size: int = 9) -> dict[tuple[int, int], Hex]:
    " Generates a hexagonal map of hexes "

    hexes: dict[tuple[int, int], Hex] = {}

    lower_bound = -(size // 2)
    upper_bound = (size // 2)

    for r in range(lower_bound, upper_bound + 1):
        qlower = max(lower_bound, lower_bound - r)
        qupper = min(upper_bound, upper_bound + -r)
        for q in range(qlower, qupper + 1):
            hexes[(q, r)] = Hex(AxialCoordinates(q, r))

    return hexes


def main() -> int:
    window = pygame.display.set_mode((WINDOW_W, WINDOW_H), pygame.DOUBLEBUF, 16)
    pygame.display.set_caption("Sam's Game")

    clock = pygame.time.Clock()
    running = True

    hexes = generate_hex_map()

    origin: Hex | None = None
    destination: Hex | None = None

    while running:
        for event in pygame.event.get():
            if _is_quit_event(event):
                running = False

            if (event.type == pygame.MOUSEBUTTONDOWN):
                if event.button == 3:  # right-click
                    for hex in hexes.values():
                        if hex.hover(offset=(WINDOW_W // 2, WINDOW_H // 2)):
                            origin = None
                            if (origin is None) or (origin != hex):
                                origin = hex

                if event.button == 1:  # left-click
                    for hex in hexes.values():
                        if hex.hover(offset=(WINDOW_W // 2, WINDOW_H // 2)):
                            hex.tile_type = TileType.WALKABLE if hex.tile_type == TileType.NON_WALKABLE else TileType.NON_WALKABLE

        destination = None
        for hex in hexes.values():
            hex.set_render_colour(BLACK if hex.tile_type == TileType.WALKABLE else GREY)
            if hex.hover(offset=(WINDOW_W // 2, WINDOW_H // 2)):
                if (hex != origin) and (hex.tile_type != TileType.NON_WALKABLE):
                    destination = hex
                elif hex.tile_type == TileType.NON_WALKABLE:
                    hex.set_render_colour(DARKER_GREY)

        path = a_star(hexes, origin, destination)

        for node in path:
            if (node == origin) or (node == destination):
                continue
            node.set_render_colour(DULL_BLUE)

        window.fill(BLACK)

        for hex in hexes.values():
            hex.render(window, offset=(WINDOW_W // 2, WINDOW_H // 2))

        pygame.display.flip()
        clock.tick(FPS)

    pygame.display.quit()
    pygame.quit()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
