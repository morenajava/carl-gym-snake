import numpy as np

from gym_snake.envs.objects import Apples
from gym_snake.envs.constants import ObjectColor
from gym_snake.envs.objects import Snake


def rotate_color(r, g, b, hue_rotation):
    if hue_rotation == 0:
        return r, g, b

    import colorsys, math
    hue, lightness, saturation = colorsys.rgb_to_hls(r / 255., g / 255., b / 255.)
    r2, g2, b2 = colorsys.hls_to_rgb((hue + hue_rotation) % 1, lightness, saturation)

    return int(math.floor(r2 * 255)), int(math.floor(g2 * 255)), int(math.floor(b2 * 255))


class BaseGrid:

    def __init__(
        self,
        np_random,
        width,
        height,
        num_snakes=1,
        initial_snake_size=4,
        num_apples=1,
        reward_apple=1,
        reward_none=0,
        reward_collision=-1,
        done_apple=False,
        always_expand=False
    ):
        assert width >= initial_snake_size
        assert height >= initial_snake_size
        assert initial_snake_size >= 2

        self.np_random = np_random
        self.num_snakes = num_snakes
        self.width = width
        self.height = height

        self.reward_apple = reward_apple
        self.reward_none = reward_none
        self.reward_collision = reward_collision

        self.done_apple = done_apple
        self.always_expand = always_expand
        self.forward_action = self.get_forward_action()

        self.snakes = None
        self.apples = Apples()
        self.all_done = False

        self.add_snakes(num_snakes, initial_snake_size)
        self.add_apples(num_apples)
        self.agent_view_size = 7
        self.grid = None
        self.init_view = None
    def move(self, actions):
        assert not self.all_done

        rewards = [self.reward_none] * self.num_snakes
        num_new_apples = 0

        # Move live snakes and eat apples
        if not self.always_expand:
            for snake, action in zip(self.snakes, actions):
                if snake.alive:
                    # Only contract if not about to eat apple
                    next_head = snake.next_head(action)
                    if next_head not in self.apples:
                        snake.contract()

        for i, snake, action in zip(range(self.num_snakes), self.snakes, actions):
            if not snake.alive:
                continue

            next_head = snake.next_head(action)
            if self.is_blocked(next_head):
                snake.kill()
                rewards[i] = self.reward_collision
            else:
                snake.expand(action)
                if next_head in self.apples:
                    if self.done_apple:
                        snake.kill()
                    self.apples.remove(next_head)
                    num_new_apples += 1
                    rewards[i] = self.reward_apple

        # If all agents are done, mark grid as done (and prevent future moves)
        dones = [not snake.alive for snake in self.snakes]
        self.all_done = False not in dones

        # Create new apples
        self.add_apples(num_new_apples)

        return rewards, dones

    def encode(self):
        return [self.encode_agent(i) for i in range(self.num_snakes)]

    def __eq__(self, other):
        self_encode = self.encode()
        other_encode = other.encode()

        if len(self_encode) != len(other_encode):
            return False

        for x, y in zip(self_encode, other_encode):
            if not np.array_equal(x, y):
                return False

        return True

    def get_forward_action(self):
        raise NotImplementedError()

    def add_snakes(self, num_snakes=1, initial_snake_size=4):
        self.snakes = []

        for i in range(num_snakes):
            x = np.random.randint(0, self.width)
            y = np.random.randint(0, self.height)
            direction = self.get_random_direction()

            rotated_green = rotate_color(0, 255, 0, i / num_snakes)
            rotated_blue = rotate_color(0, 0, 255, i / num_snakes)

            new_snake = Snake(x, y, direction, color_head=rotated_blue, color_body=rotated_green)
            self.snakes.append(new_snake)
            for _ in range(initial_snake_size):
                next_head = new_snake.next_head(self.forward_action)
                if self.is_blocked(next_head):
                    # give up and try again to place snakes
                    return self.add_snakes(num_snakes=num_snakes, initial_snake_size=initial_snake_size)

                new_snake.expand(self.forward_action)

    def add_apples(self, num_apples):
        num_open_spaces = self.width * self.height - sum(len(s) for s in self.snakes) - len(self.apples)
        num_new_apples = min(num_apples, num_open_spaces)
        for _ in range(num_new_apples):
            self._add_one_apple()

    def _add_one_apple(self):
        while True:
            p = (self.np_random.randint(0, self.width), self.np_random.randint(0, self.height))
            if self.is_blocked(p) or p in self.apples:
                continue

            self.apples.add(p)
            break

    def is_blocked(self, p):
        x, y = p
        if x < 0 or x >= self.width:
            return True
        if y < 0 or y >= self.height:
            return True

        for snake in self.snakes:
            if p in snake:
                return True

        return False

    def encode_agent(self, agent_number):
        result = np.zeros((self.width, self.height, 3), dtype='uint8')
        snake_v = None

        for col in range(result.shape[0]):
            for row in range(result.shape[1]):
             result[col][row] = ObjectColor.empty

        for p in self.apples:
            result[p] = ObjectColor.apple

        for i, snake in enumerate(self.snakes):
            if not snake.alive:
                body_color = ObjectColor.dead_body
                head_color = ObjectColor.dead_head
            elif i == agent_number:
                body_color = ObjectColor.own_body
                head_color = ObjectColor.own_head
                print("snake direction: ", snake._direction)
                snake_v = snake
            else:
                body_color = ObjectColor.other_body
                head_color = ObjectColor.other_head

            last_p = None
            for p in snake:
                result[p] = body_color
                last_p = p

            result[last_p] = head_color
        self.init_view = result
        self.grid, vis_mask = self.gen_obs_grid(snake_v)
        return self.grid.encode(vis_mask)
        # return result

    # Morena
    def get_view_exts(self,snake):
        """
        Get the extents of the square set of tiles visible to the agent
        Note: the bottom extent indices are not included in the set
        """

        # Facing right
        if snake._direction == 1:
            topX = snake.snake_head()[0]
            topY = snake.snake_head()[1] - self.height // 2
        # Facing down
        elif snake._direction == 2:
            topX = snake.snake_head()[0] - self.agent_view_size // 2
            topY = snake.snake_head()[1]
        # Facing left
        elif snake._direction == 3:
            topX = snake.snake_head()[0] - self.agent_view_size + 1
            topY = snake.snake_head()[1] - self.agent_view_size // 2
        # Facing up
        elif snake._direction == 0:
            topX = snake.snake_head()[0] - self.agent_view_size // 2
            topY = snake.snake_head()[1] - self.agent_view_size + 1
        else:
            assert False, "invalid agent direction"

        botX = topX + self.agent_view_size
        botY = topY + self.agent_view_size

        return (topX, topY, botX, botY)

    # Morena
    def gen_obs_grid(self,snake):
        """
        Generate the sub-grid observed by the agent.
        This method also outputs a visibility mask telling us which grid
        cells the agent can actually see.
        """

        topX, topY, botX, botY = self.get_view_exts(snake)

        grid = self.slice(topX, topY, self.agent_view_size, self.agent_view_size)

        for i in range(4 - snake._direction):
            grid = grid.rotate_left()

        vis_mask = np.ones(shape=(grid.width, grid.height), dtype=bool)

        # Make it so the agent sees what it's carrying
        # We do this by placing the carried object at the agent's position
        # in the agent's partially observable view
        agent_pos = grid.width // 2, grid.height - 1

        # grid.set(*agent_pos, None)

        return grid, vis_mask
        # Morena
    def get(self, i, j):
        assert i >= 0 and i < self.width
        assert j >= 0 and j < self.height
        return self.grid[j * self.width + i]

    def get_random_direction(self):
        raise NotImplementedError()

    def get_renderer_dimensions(self, tile_size):
        raise NotImplementedError()

    def render(self, r, tile_size, cell_pixels):
        raise NotImplementedError()
    def slice(self, topX, topY, width, height):
            """
            Get a subset of the grid
            """

            grid = Grid(width, height)

            for j in range(0, height):
                for i in range(0, width):
                    x = topX + i
                    y = topY + j

                    if x >= 0 and x < self.width and \
                            y >= 0 and y < self.height:
                        v = self.init_view[x][y]
                    else:
                        v = ObjectColor.wall

                    grid.set(i, j, v)

            return grid
class Grid:
        """
        Represent a grid and operations on it
        """

        # Static cache of pre-renderer tiles
        tile_cache = {}

        def __init__(self, width, height):
            assert width >= 3
            assert height >= 3

            self.width = width
            self.height = height

            self.grid = [None] * width * height

        # def __contains__(self, key):
        #     if isinstance(key, WorldObj):
        #         for e in self.grid:
        #             if e is key:
        #                 return True
        #     elif isinstance(key, tuple):
        #         for e in self.grid:
        #             if e is None:
        #                 continue
        #             if (e.color, e.type) == key:
        #                 return True
        #             if key[0] is None and key[1] == e.type:
        #                 return True
        #     return False

        def __eq__(self, other):
            grid1 = self.encode()
            grid2 = other.encode()
            return np.array_equal(grid2, grid1)

        def __ne__(self, other):
            return not self == other

        def copy(self):
            from copy import deepcopy
            return deepcopy(self)

        def set(self, i, j, v):
            assert i >= 0 and i < self.width
            assert j >= 0 and j < self.height
            self.grid[j * self.width + i] = v

        def get(self, i, j):
            assert i >= 0 and i < self.width
            assert j >= 0 and j < self.height
            return self.grid[j * self.width + i]
            # return self.grid[i * self.height + j]

        # def horz_wall(self, x, y, length=None, obj_type=Wall):
        #     if length is None:
        #         length = self.width - x
        #     for i in range(0, length):
        #         self.set(x + i, y, obj_type())
        #
        # def vert_wall(self, x, y, length=None, obj_type=Wall):
        #     if length is None:
        #         length = self.height - y
        #     for j in range(0, length):
        #         self.set(x, y + j, obj_type())

        def wall_rect(self, x, y, w, h):
            self.horz_wall(x, y, w)
            self.horz_wall(x, y + h - 1, w)
            self.vert_wall(x, y, h)
            self.vert_wall(x + w - 1, y, h)

        def rotate_left(self):
            """
            Rotate the grid to the left (counter-clockwise)
            """

            grid = Grid(self.height, self.width)

            for i in range(self.width):
                for j in range(self.height):
                    v = self.get(i, j)
                    grid.set(j, grid.height - 1 - i, v)

            return grid

        # def slice(self, topX, topY, width, height):
        #     """
        #     Get a subset of the grid
        #     """
        #
        #     grid = Grid(width, height)
        #
        #     for j in range(0, height):
        #         for i in range(0, width):
        #             x = topX + i
        #             y = topY + j
        #
        #             if x >= 0 and x < self.width and \
        #                     y >= 0 and y < self.height:
        #                 v = self.get(x, y)
        #             else:
        #                 v = Wall()
        #
        #             grid.set(i, j, v)
        #
        #     return grid

        # @classmethod
        # def render_tile(
        #         cls,
        #         obj,
        #         agent_dir=None,
        #         highlight=False,
        #         tile_size=TILE_PIXELS,
        #         subdivs=3
        # ):
        #     """
        #     Render a tile and cache the result
        #     """
        #
        #     # Hash map lookup key for the cache
        #     key = (agent_dir, highlight, tile_size)
        #     key = obj.encode() + key if obj else key
        #
        #     if key in cls.tile_cache:
        #         return cls.tile_cache[key]
        #
        #     img = np.zeros(shape=(tile_size * subdivs, tile_size * subdivs, 3), dtype=np.uint8)
        #
        #     # Draw the grid lines (top and left edges)
        #     # fill_coords(img, point_in_rect(0, 0.031, 0, 1), (100, 100, 100))
        #     # fill_coords(img, point_in_rect(0, 1, 0, 0.031), (100, 100, 100))
        #
        #     if obj != None:
        #         obj.render(img)
        #
        #     # Overlay the agent on top
        #     if agent_dir is not None:
        #         tri_fn = point_in_triangle(
        #             (0.12, 0.19),
        #             (0.87, 0.50),
        #             (0.12, 0.81),
        #         )
        #
        #         # Rotate the agent based on its direction
        #         tri_fn = rotate_fn(tri_fn, cx=0.5, cy=0.5, theta=0.5 * math.pi * agent_dir)
        #         fill_coords(img, tri_fn, (255, 0, 0))
        #
        #     # Highlight the cell if needed
        #     if highlight:
        #         highlight_img(img)
        #
        #     # Downsample the image to perform supersampling/anti-aliasing
        #     img = downsample(img, subdivs)
        #
        #     # Cache the rendered tile
        #     cls.tile_cache[key] = img
        #
        #     return img

        # def render(
        #         self,
        #         tile_size,
        #         agent_pos=None,
        #         agent_dir=None,
        #         highlight_mask=None
        # ):
        #     """
        #     Render this grid at a given scale
        #     :param r: target renderer object
        #     :param tile_size: tile size in pixels
        #     """
        #
        #     if highlight_mask is None:
        #         highlight_mask = np.zeros(shape=(self.width, self.height), dtype=bool)
        #
        #     # Compute the total grid size
        #     width_px = self.width * tile_size
        #     height_px = self.height * tile_size
        #
        #     img = np.zeros(shape=(height_px, width_px, 3), dtype=np.uint8)
        #
        #     # Render the grid
        #     for j in range(0, self.height):
        #         for i in range(0, self.width):
        #             cell = self.get(i, j)
        #
        #             agent_here = np.array_equal(agent_pos, (i, j))
        #             tile_img = Grid.render_tile(
        #                 cell,
        #                 agent_dir=agent_dir if agent_here else None,
        #                 highlight=highlight_mask[i, j],
        #                 tile_size=tile_size
        #             )
        #
        #             ymin = j * tile_size
        #             ymax = (j + 1) * tile_size
        #             xmin = i * tile_size
        #             xmax = (i + 1) * tile_size
        #             img[ymin:ymax, xmin:xmax, :] = tile_img
        #
        #     return img

        def encode(self, vis_mask=None):
            """
            Produce a compact numpy encoding of the grid
            """

            if vis_mask is None:
                vis_mask = np.ones((self.width, self.height), dtype=bool)

            array = np.zeros((self.width, self.height, 3), dtype='uint8')

            for i in range(self.width):
                for j in range(self.height):
                    if vis_mask[i, j]:
                        v = self.get(i, j)

                        if v is None:
                            array[i, j] = ObjectColor.empty  # modified morena

                        else:
                            array[i, j, :] = v

            return array

        @staticmethod
        def decode(array):
            """
            Decode an array grid encoding back into a grid
            """

            width, height, channels = array.shape
            assert channels == 3

            vis_mask = np.ones(shape=(width, height), dtype=bool)

            grid = Grid(width, height)
            for i in range(width):
                for j in range(height):
                    # type_idx, color_idx, state = array[i, j]
                    # morena
                    state = array[i, j]
                    # v = WorldObj.decode(type_idx, color_idx, state)
                    # grid.set(i, j, v)
                    grid.set(i, j, state)

                    # vis_mask[i, j] = (type_idx != OBJECT_TO_IDX['unseen'])

            # return grid, vis_mask

            return grid

        def process_vis(grid, agent_pos):
            mask = np.zeros(shape=(grid.width, grid.height), dtype=bool)

            mask[agent_pos[0], agent_pos[1]] = True

            for j in reversed(range(0, grid.height)):
                for i in range(0, grid.width - 1):
                    if not mask[i, j]:
                        continue

                    cell = grid.get(i, j)
                    if cell and not cell.see_behind():
                        continue

                    mask[i + 1, j] = True
                    if j > 0:
                        mask[i + 1, j - 1] = True
                        mask[i, j - 1] = True

                for i in reversed(range(1, grid.width)):
                    if not mask[i, j]:
                        continue

                    cell = grid.get(i, j)
                    if cell and not cell.see_behind():
                        continue

                    mask[i - 1, j] = True
                    if j > 0:
                        mask[i - 1, j - 1] = True
                        mask[i, j - 1] = True

            for j in range(0, grid.height):
                for i in range(0, grid.width):
                    if not mask[i, j]:
                        grid.set(i, j, None)

            return mask