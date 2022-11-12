import pygame as pg
import pygame.draw
import numpy as np
from scipy.cluster.vq import vq, kmeans, whiten
from scipy.spatial import KDTree
import statistics


class Point:
    def __init__(self, pos):
        """
        :param pos: tuple(x, y)
        """
        self._pos = pos

    def get_position(self):
        return self._pos

    @property
    def x(self):
        if self._pos:
            return self._pos[0]

    @property
    def y(self):
        if self._pos:
            return self._pos[1]


def distance_between_points(first, second):
    return (((first[0] - second[0]) ** 2) + ((first[1] - second[1]) ** 2)) ** 0.5


def draw_point(pos, screen, color=(0, 255, 0), radius: int = 7, width: int = 2):
    pygame.draw.circle(
        screen,
        color,
        pos,
        radius,
        width
    )


class PointPygame(Point):
    def __init__(self, pos, screen, color=(0, 255, 0), radius: int = 7, width: int = 2):
        super().__init__(pos=pos)
        self._screen = screen
        self._color = color
        self._width = width
        self._radius = radius

    def distance(self, other: Point) -> float:
        return distance_between_points(self._pos, other.get_position())

    def update_pos(self, new_pos):
        self._pos = new_pos

    def draw(self):
        if self._pos is not None:
            pygame.draw.circle(
                self._screen,
                self._color,
                self._pos,
                self._radius,
                self._width
            )

    def __str__(self):
        pos = 'None'

        if self._pos is not None:
            pos = f'x: {self._pos[0]}, y: {self._pos[1]}'
        return f'{self.__class__}: pos: {pos}'


class Circle:
    def __init__(self, init_pos, radius):
        self._pos = init_pos
        self._radius = radius
        self._radius_sq = radius * radius

    @property
    def x(self):
        return self._pos[0]

    @property
    def y(self):
        return self._pos[1]

    def is_inside_circle(self, pos):
        distance = distance_between_points(self._pos, pos)
        return distance <= self._radius

    def update_pos(self, new_pos):
        self._pos = new_pos

    def update_radius(self, value, correction_func=lambda x: x):
        """
        :param value: some value (example: time delta)
        :param correction_func: function that can transform value to distance
        """
        self._radius = correction_func(value)

    def get_intersection_with_circle(self, other: "Circle") -> ((float, float), (float, float)):
        """
        https://math.stackexchange.com/questions/256100/how-can-i-find-the-points-at-which-two-circles-intersect
        :param other: 
        :return: 
        """
        centers_distance = distance_between_points(first=self._pos, second=other._pos)
        if centers_distance <= 0.001:
            return None, None

        q = (self._radius ** 2 - other._radius ** 2) / (2 * centers_distance ** 2)

        w0 = (self._radius ** 2 + other._radius ** 2) / centers_distance ** 2
        w1 = (self._radius ** 2 - other._radius ** 2) ** 2 / centers_distance ** 4
        w2 = 2 * w0 - w1 - 1

        if w2 < 0:
            return None, None

        w = 0.5 * w2 ** 0.5
        px1 = 0.5 * (self.x + other.x) + q * (other.x - self.x)
        py1 = 0.5 * (self.y + other.y) + q * (other.y - self.y)
        px2 = other.y - self.y
        py2 = self.x - other.x

        x1 = px1 + w * px2
        y1 = py1 + w * py2

        x2 = px1 - w * px2
        y2 = py1 - w * py2

        return (x1, y1), (x2, y2)


class CirclePygame(Circle):
    def __init__(self, screen, color, init_pos, radius, width):
        super().__init__(init_pos=init_pos, radius=radius)
        self._screen = screen
        self._color = color
        self._width = width

        self._current_width = 5
        self._current_color = (255, 0, 0)
        self._is_current = False
        self._is_moving = False

    def set_current(self):
        self._is_current = True

    def unset_current(self):
        self._is_current = False

    def is_current(self):
        return self._is_current

    def increase_size(self, amount: int = 5):
        self._radius += amount

    def decrease_size(self, amount: int = 5):
        self._radius = max(amount, self._radius - amount)

    def draw(self):
        if self._is_current:
            pygame.draw.circle(
                self._screen,
                self._current_color,
                self._pos,
                self._radius,
                self._current_width
            )
        else:
            pygame.draw.circle(
                self._screen,
                self._color,
                self._pos,
                self._radius,
                self._width
            )


def handle_events(antenas):
    running = True

    for e in pg.event.get():
        click_pos = pg.mouse.get_pos()
        if e.type == pg.MOUSEBUTTONDOWN:

            for a in antenas:
                if a.is_inside_circle(pos=click_pos):
                    a.set_current()
                    break

        if e.type == pg.MOUSEBUTTONUP:
            for a in antenas:
                a.unset_current()

        if e.type == pygame.KEYDOWN:
            if e.key == pg.K_a and click_pos:
                new_circle = default_circle_args.copy()
                new_circle["init_pos"] = click_pos
                antenas.append(
                    CirclePygame(**new_circle)
                )

            if e.key == pg.K_PLUS or e.key == pg.K_KP_PLUS:
                for a in antenas:
                    if a.is_current():
                        a.increase_size()
            if e.key == pg.K_MINUS or e.key == pg.K_KP_MINUS:
                for a in antenas:
                    if a.is_current():
                        a.decrease_size()

        if e.type == pg.QUIT:
            running = False
    return running


def update_objects(antenas, points, location, median):
    for a in antenas:
        if a.is_current():
            click_pos = pg.mouse.get_pos()
            a.update_pos(click_pos)

    for a in antenas:
        a.draw()

    # intersection points
    for p in points:
        p.draw()

    if location.get_position():
        window.blit(kd_tree_text, (location.x - kd_tree_text.get_width() - 10, location.y))
        location.draw()
    if median.get_position():
        window.blit(median_text, (median.x + 10, median.y - median_text.get_height()))
        median.draw()


pg.init()
pg.font.init()
WIDTH = 640
HEIGHT = 640
window = pg.display.set_mode((WIDTH, HEIGHT))
window.fill((255, 255, 255))
default_circle_args = {"screen": window, "color": (0, 0, 0), "init_pos": (100, 100), "radius": 25, "width": 1}

my_font = pg.font.SysFont('Arial', 15)
kd_tree_text = my_font.render('kd_tree', True, (0, 0, 0))
median_text = my_font.render('median', True, (0, 0, 0))


def main():
    running = True
    clock = pg.time.Clock()

    antenas: list[CirclePygame] = [
        CirclePygame(**{"screen": window, "color": (0, 0, 0), "init_pos": (150, 400), "radius": 150, "width": 1}),
        CirclePygame(**{"screen": window, "color": (0, 0, 0), "init_pos": (320, 200), "radius": 150, "width": 1}),
        CirclePygame(**{"screen": window, "color": (0, 0, 0), "init_pos": (450, 450), "radius": 180, "width": 1}),
    ]

    points = []
    location = PointPygame(pos=None, screen=window, color=(0, 0, 255))

    median = PointPygame(pos=None, screen=window, color=(0, 0, 255))

    while running:
        clock.tick(60)
        window.fill((255, 255, 255))
        location.update_pos(None)

        running = handle_events(antenas=antenas)
        count = 0
        x_ = []
        y_ = []
        p_ = []
        if len(antenas) > 1:
            # adding intersection points to array (begin)
            num = len(antenas) * len(antenas) - 1
            if len(points) <= num:
                for _ in range(num - len(points)):
                    points.append(PointPygame(pos=None, screen=default_circle_args['screen']))
                    x_.append(None)
                    y_.append(None)
            # adding intersection points to array (end)

            for i in range(len(antenas)):
                for j in range(i + 1, len(antenas)):
                    point_0, point_1 = antenas[i].get_intersection_with_circle(antenas[j])

                    points[count].update_pos(point_0)
                    count += 1
                    points[count].update_pos(point_1)
                    count += 1
                    if point_0:
                        x_.append(point_0[0])
                        x_.append(point_1[0])
                        y_.append(point_0[1])
                        y_.append(point_1[1])
                        p_.append(point_0)
                        p_.append(point_1)

            # kd tree
            if any(x_):
                p_temp_ = [x for x in p_ if x is not None]
                tree = KDTree(p_temp_)
                result = tree.query_ball_point(p_temp_, r=100)
                result_max = result[0]
                for s in result:
                    if len(result_max) < len(s):
                        result_max = s
                qw = np.array([p_temp_[x] for x in result_max])
                # print(qw)
                x_median = statistics.median(qw[:, 0])
                y_median = statistics.median(qw[:, 1])
                # median.update_pos((x_median, y_median))
                location.update_pos((x_median, y_median))

            # using median
            if any(x_):
                x_temp_ = [x for x in x_ if x is not None]
                y_temp_ = [y for y in y_ if y is not None]
                x_median = statistics.median(x_temp_)
                y_median = statistics.median(y_temp_)
            #     x_mean = statistics.mean(x_temp_)
            #     y_mean = statistics.mean(y_temp_)
            #     x_mode = statistics.mode([round(x, -1) for x in x_temp_])
            #     y_mode = statistics.mode([round(x, -1) for x in y_temp_])
            #
                l_w = 2
            #     pygame.draw.line(window, (255, 0, 0), (x_mean, 0), (x_mean, HEIGHT), width=l_w)
            #     pygame.draw.line(window, (255, 0, 0), (0, y_mean), (WIDTH, y_mean), width=l_w)
            #     pygame.draw.line(window, (0, 255, 0), (x_mode, 0), (x_mode, HEIGHT), width=l_w)
            #     pygame.draw.line(window, (0, 255, 0), (0, y_mode), (WIDTH, y_mode), width=l_w)
            #     pygame.draw.line(window, (0, 0, 255), (x_median, 0), (x_median, HEIGHT), width=l_w)
            #     pygame.draw.line(window, (0, 0, 255), (0, y_median), (WIDTH, y_median), width=l_w)
                median.update_pos((x_median, y_median))

            # sliding window
            # if any(x_):
            #     x_temp = sorted([x for x in x_ if x is not None])
            #     y_temp = sorted([y for y in y_ if y is not None])
            #     x_range = max(x_temp) - min(x_temp)
            #     y_range = max(y_temp) - min(y_temp)
            #     x_window_size = x_range / 3
            #     y_window_size = y_range / 3
            #     result_x = []
            #     result_y = []
            #     temp_len_x = 0
            #     temp_len_y = 0
            #
            #     for _x in x_temp:
            #         temp_x_window = [i for i, x in enumerate(x_temp) if x_window_size + _x > x >= _x]
            #         if len(temp_x_window) > temp_len_x:
            #             temp_len_x = len(temp_x_window)
            #             result_x = temp_x_window
            #     y_temp = [__y for i, __y in enumerate(y_temp) if i in result_x]
            #     for _y in y_temp:
            #         temp_y_window = [i for i, y in enumerate(y_temp) if y_window_size + _y > y >= _y]
            #         if len(temp_y_window) > temp_len_y:
            #             temp_len_y = len(temp_y_window)
            #             result_y = temp_y_window
            #
            #     if result_x and result_y:
            #         result_x = [__x for i, __x in enumerate(x_temp) if i in result_x]
            #         result_y = [__y for i, __y in enumerate(y_temp) if i in result_y]
            #
            #         pygame.draw.line(window, (0, 0, 0), (min(result_x), 0), (min(result_x), HEIGHT), width=5)
            #         pygame.draw.line(window, (0, 0, 0), (max(result_x), 0), (max(result_x), HEIGHT), width=5)
            #         pygame.draw.line(window, (0, 0, 0), (0, min(result_y)), (WIDTH, min(result_y)), width=5)
            #         pygame.draw.line(window, (0, 0, 0), (0, max(result_y)), (WIDTH, max(result_y)), width=5)
            #
            #         location.update_pos((sum(result_x) / len(result_x), sum(result_y) / len(result_y)))

        update_objects(antenas=antenas, points=points, location=location, median=median)

        pg.display.update()

    pg.quit()


if __name__ == "__main__":
    main()
