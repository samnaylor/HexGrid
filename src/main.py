# TODO: Some sort of global game state manager - a singleton object that manages everything

from math import sqrt, sin, cos, radians
from typing import Generic, TypeVar, Any
from dataclasses import dataclass

import pygame

pygame.init()

hex_size = 30
hex_width = sqrt(3) * hex_size
hex_height = hex_size * 3 / 2
hex_spacing_horizontal = hex_width
hex_spacing_vertical = hex_height

camera_move_speed = 20
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


@dataclass(eq=True, unsafe_hash=True)
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

    def is_hover(self) -> bool:
        return self.coords.pixel_to_hex(pygame.mouse.get_pos()) == self.coords

    def vertices(self) -> list[tuple[float, float]]:
        centerx, centery = self.coords.to_pixel()
        verts: list[tuple[float, float]] = []

        for i in range(6):
            angle = radians(60 * i - 30)
            verts.append((centerx + hex_size * cos(angle), centery + hex_size * sin(angle)))

        return verts

    def reachable(self, movement: int = 1) -> set[Axial]:
        " Tiles reachable from here within that movement "

        state = Game()

        visited = set[Axial]()
        visited.add(self.coords)
        fringes: list[list[Axial]] = [[self.coords]]

        for k in range(1, movement + 1):
            fringes.append([])
            for hex in fringes[k - 1]:
                for direction in range(6):
                    neighbour = hex.neighbour(direction)
                    if (neighbour not in visited) and (not state.store[(int(neighbour.q), int(neighbour.r))].obstacle):
                        visited.add(neighbour)
                        fringes[k].append(neighbour)

        return visited

    def render(self, surface: pygame.Surface) -> None:
        colour = self.hover_colour if self.is_hover() else self.base_colour
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

    def add_offset(self, delta: tuple[float, float]) -> None:
        self.offset = (self.offset[0] + delta[0], self.offset[1] + delta[1])
        print(self.offset)


def is_quit_event(event: pygame.event.Event) -> bool:
    return (event.type == pygame.QUIT) or ((event.type == pygame.KEYDOWN) and (event.key == pygame.K_ESCAPE))


def main() -> int:
    screen = pygame.display.set_mode((640, 480))
    clock = pygame.time.Clock()
    running = True

    black = (0, 0, 0)
    white = (255, 255, 255)

    state = Game()
    hovered: HexTile | None = None

    while running:
        for event in pygame.event.get():
            if is_quit_event(event):
                running = False

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

        if hovered is not None:
            line_from_origin = state.store[(0, 0)].coords.line(hovered.coords)
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
