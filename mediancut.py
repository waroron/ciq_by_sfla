from PIL import Image
import numpy as np
from cube import ColorCube


def get_colors(image):
    colors = image.getcolors(image.size[0] * image.size[1])
    return [c[1] for c in colors]


def median_cut(image, num_colors):
    colors = get_colors(image)
    cubes = [ColorCube(*colors)]
    while len(cubes) < num_colors:
        global_max_size = 0

        for index, cube in enumerate(cubes):
            size = cube.size
            max_size = max(size)
            max_dim = size.index(max_size)

            if max_size > global_max_size:
                global_max_size = max_size
                max_cube = index

        split_box = cubes[max_cube]
        cube_a, cube_b = split_box.split(max_dim)
        cubes = cubes[:max_cube] + [cube_a, cube_b] + cubes[max_cube + 1:]
    return np.array([c.average for c in cubes]).astype(np.int), cubes


if __name__ == '__main__':
    img_name = "pills.png"
    img = Image.open(img_name)
    img = median_cut(img, 16)

    img.show()
