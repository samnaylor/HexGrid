from math import sqrt, sin, cos, radians
from heapq import heappop, heappush
from typing import Generic, TypeVar, Any
from dataclasses import dataclass

import pygame

pygame.init()

hex_size = 30
hex_width = sqrt(3) * hex_size
hex_height = hex_size * 3 / 2
hex_spacing_horizontal = hex_width
hex_spacing_vertical = hex_height

camera_move_speed = 10
world_width = 1280
world_height = 960


T = TypeVar("T")


class Singleton(Generic[T], type):
    def __init__(self: "Singleton[T]", __name: str, __bases: tuple[type, ...], __dict: dict[str, Any]) -> None:
        super(Singleton, self).__init__(__name, __bases, __dict)
        self._instance: T = super(Singleton, self).__call__()

    def __call__(self: "Singleton[T]", *args: Any, **kwargs: Any) -> T:
        return self._instance


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


@dataclass(eq=True, order=True, unsafe_hash=True)
class Axial:
    " axial coordinates "

    q: float
    r: float

    def to_cube(self) -> tuple[float, float, float]:
        return (self.q, self.r, -(self.q + self.r))

    @staticmethod
    def round(axial: "Axial") -> "Axial":
        cube = axial.to_cube()

        q = round(cube[0])
        r = round(cube[1])
        s = round(cube[2])

        dq = abs(q - cube[0])
        dr = abs(r - cube[1])
        ds = abs(s - cube[2])

        if (dq > dr) and (dq > ds):
            q = -(r + s)
        elif (dr > ds):
            r = -(q + s)

        return Axial(q, r)

    @staticmethod
    def pixel_to_hex(point: tuple[float, float]) -> "Axial":
        offset = Game().offset
        px, py = (point[0] + offset[0]), (point[1] + offset[1])
        q = (sqrt(3) / 3 * px - 1 / 3 * py) / hex_size
        r = (2 / 3 * py) / hex_size

        return Axial.round(Axial(q, r))

    @staticmethod
    def direction(i: int) -> "Axial":
        return [Axial(1, 0), Axial(1, -1), Axial(0, -1), Axial(-1, 0), Axial(-1, 1), Axial(0, 1)][i]

    def add(self, other: "Axial") -> "Axial":
        return Axial(self.q + other.q, self.r + other.r)

    def sub(self, other: "Axial") -> "Axial":
        return Axial(self.q - other.q, self.r - other.r)

    def neighbour(self, i: int) -> "Axial":
        return self.add(Axial.direction(i))

    def distance(self, other: "Axial") -> float:
        diff = self.sub(other)
        return (abs(diff.q) + abs(diff.q + diff.r) + abs(diff.r)) / 2

    def lerp(self, other: "Axial", t: float) -> "Axial":
        return Axial(lerp(self.q, other.q, t), lerp(self.r, other.r, t))

    def line(self, other: "Axial") -> list["Axial"]:
        N = self.distance(other)
        axials: list[Axial] = []

        for i in range(0, round(N)):
            axials.append(Axial.round(self.lerp(other, 1.0 / N * i)))

        axials.append(other)

        return axials

    def to_pixel(self) -> tuple[float, float]:
        ox, oy = Game().offset

        x = hex_size * (sqrt(3) * self.q + sqrt(3) / 2 * self.r) - ox
        y = hex_size * (3 / 2 * self.r) - oy

        return (x, y)


@dataclass(unsafe_hash=True)
class HexTile:
    coords: Axial
    obstacle: bool = False

    base_colour: tuple[int, int, int] = (0, 0, 0)
    hover_colour: tuple[int, int, int] = (0, 255, 0)
    border_colour: tuple[int, int, int] = (255, 255, 255)
    obstacle_colour: tuple[int, int, int] = (255, 0, 0)

    @property
    def cost(self) -> int:
        return 999 if self.obstacle else 1

    def is_hover(self) -> bool:
        return self.coords.pixel_to_hex(pygame.mouse.get_pos()) == self.coords

    def vertices(self) -> list[tuple[float, float]]:
        centerx, centery = self.coords.to_pixel()
        verts: list[tuple[float, float]] = []

        for i in range(6):
            angle = radians(60 * i - 30)
            verts.append((centerx + hex_size * cos(angle), centery + hex_size * sin(angle)))

        return verts

    def reachable(self, goal: Axial, movement: int = 10) -> bool:
        " Tiles reachable from here within movement tiles "

        state = Game()

        visited = set[Axial]()
        visited.add(self.coords)
        fringes: list[list[Axial]] = [[self.coords]]

        for k in range(1, movement + 1):
            fringes.append([])
            for hex in fringes[k - 1]:
                for direction in range(6):
                    neighbour = hex.neighbour(direction)
                    exists = state.has_tile(neighbour)
                    if exists and ((neighbour not in visited) and (not state.get_tile(neighbour).obstacle)):
                        visited.add(neighbour)
                        fringes[k].append(neighbour)

        return goal.coords in visited

    def toggle_obstacle(self) -> None:
        self.obstacle = not self.obstacle

    def render(self, surface: pygame.Surface) -> None:
        colour = self.hover_colour if self.is_hover() else self.base_colour
        colour = colour if not self.obstacle else self.obstacle_colour
        verts = self.vertices()

        pygame.draw.polygon(surface, colour, verts)
        pygame.draw.polygon(surface, self.border_colour, verts, 2)


class Game(metaclass=Singleton):
    def __init__(self) -> None:
        self.offset: tuple[float, float] = (-(world_width // 2), -(world_height // 2))
        self.store: dict[tuple[int, int], HexTile] = {}

        locations: list[tuple[int, int]] = [
            (0, -3), (1, -3), (2, -3), (3, -3),
            (-1, -2), (0, -2), (1, -2), (2, -2), (3, -2),
            (-2, -1), (-1, -1), (0, -1), (1, -1), (2, -1), (3, -1),
            (-3, 0), (-2, 0), (-1, 0), (0, 0), (1, 0), (2, 0), (3, 0),
            (-3, 1), (-2, 1), (-1, 1), (0, 1), (1, 1), (2, 1),
            (-3, 2), (-2, 2), (-1, 2), (0, 2), (1, 2),
            (-3, 3), (-2, 3), (-1, 3), (0, 3)
        ]

        for (q, r) in locations:
            self.store[(q, r)] = HexTile(Axial(q, r))

    def has_tile(self, coords: Axial) -> bool:
        return self.store.get((coords.q, coords.r)) is not None

    def get_tile(self, coords: Axial) -> HexTile:
        return self.store[(coords.q, coords.r)]

    def add_offset(self, delta: tuple[float, float]) -> None:
        self.offset = (self.offset[0] + delta[0], self.offset[1] + delta[1])


def astar_pathfinding(start: Axial, goal: Axial) -> list[Axial]:
    def heuristic(a: Axial, b: Axial) -> float:
        return a.distance(b)

    def get_neighbours(tile: Axial) -> list[Axial]:  # TODO: filter out not real tiles here
        return [tile.neighbour(i) for i in range(6)]

    state = Game()
    frontier = [(0, start)]
    came_from = {}
    cost_so_far = {start: 0}

    while frontier:
        _, current = heappop(frontier)

        if current == goal:
            break

        for neighbour in get_neighbours(current):
            if state.has_tile(neighbour):
                new_cost = cost_so_far[current] + state.get_tile(neighbour).cost

                if (neighbour not in cost_so_far) or (new_cost < cost_so_far[neighbour]):
                    cost_so_far[neighbour] = new_cost
                    priority = new_cost + heuristic(goal, neighbour)
                    heappush(frontier, (priority, neighbour))
                    came_from[neighbour] = current

    path = [goal]
    while path[-1] != start:
        path.append(came_from[path[-1]])
    path.reverse()

    return path


def is_quit_event(event: pygame.event.Event) -> bool:
    return (event.type == pygame.QUIT) or ((event.type == pygame.KEYDOWN) and (event.key == pygame.K_ESCAPE))


def main() -> int:
    screen = pygame.display.set_mode((640, 480))
    clock = pygame.time.Clock()
    running = True

    black = (0, 0, 0)
    white = (255, 255, 255)

    state = Game()
    origin = state.get_tile(Axial(0, 0))
    hovered: HexTile | None = None

    while running:
        for event in pygame.event.get():
            if is_quit_event(event):
                running = False
            
            if event.type == pygame.MOUSEBUTTONDOWN:
                tile = Axial.pixel_to_hex(event.pos)
                
                if state.has_tile(tile):
                    state.get_tile(tile).toggle_obstacle()

        screen.fill(black)

        pressed = pygame.key.get_pressed()
        dx = camera_move_speed * (pressed[pygame.K_d] - pressed[pygame.K_a])
        dy = camera_move_speed * (pressed[pygame.K_s] - pressed[pygame.K_w])

        state.add_offset((dx, dy))

        hovered = None
        for (_, tile) in state.store.items():
            tile.render(screen)

            if tile.is_hover():
                hovered = tile

        if (hovered is not None) and (not hovered.obstacle) and (origin.reachable(hovered)):
            line_from_origin = astar_pathfinding(state.store[(0, 0)].coords, hovered.coords)
            points: list[tuple[float, float]] = []

            for hex in line_from_origin:
                points.append(hex.to_pixel())

            if len(points) >= 2:
                pygame.draw.lines(screen, white, False, points, 3)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
