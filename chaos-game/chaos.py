import itertools
import argparse
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm


def parse_points(vertices, partitions, resolution=10):
    """Extract user-selected vertices for each shape specified
    """
    plt.imshow(np.zeros((resolution, resolution)), cmap='gray')
    plt.axis("off")

    print(f"Select{','.join(f' a {p}-sided shape' for p in partitions)}, with each non-intersecting and non-collinear")
    vertices = plt.ginput(n=vertices)
    plt.close("all")

    shapes = {
        i: vertices[j: j + partitions[i]]
        for i, j in enumerate(([0] + partitions)[:-1])
    }
    return shapes


def chaos_game(shapes, iterations, resolution=10):
    """Run the chaos game for each shape, and randomly switch between shapes w/ equal prob.
    """
    # pick random initial point in the plane
    x, y = resolution * np.random.rand(2)

    jump = 0.5
    vert_ind = -1

    point_buffer = [[x, y]]
    rand_shape = np.random.choice(len(shapes))
    for i in tqdm(range(iterations), desc="Running the chaos game"):
        vertices = shapes[rand_shape]
        rand_vert_ind = np.random.choice(len(vertices))

        # if not a triangle, apply a special rule
        if len(vertices) > 3:
            while rand_vert_ind == vert_ind:
                rand_vert_ind = np.random.choice(len(vertices))

        vert_ind = rand_vert_ind
        rand_vertex = vertices[vert_ind]

        # generate the next point
        x, y = jump * (x + rand_vertex[0]), jump * (y + rand_vertex[1])
        point_buffer.append([x, y])

        # randomly switch to next shape
        rand_shape = np.random.choice(len(shapes))

    return np.array(point_buffer)


def plot_results(shapes, points, transient=100):
    """Plots the shapes and the results from running the chaos game
    """
    plt.style.use('dark_background')
    plt.figure()
    plt.axis("off")

    # first plot the chaos game points, also skip the first transient points for a nice plot
    plt.scatter(*points[transient:].T, s=1e-2)

    # plot shape vertices
    for s in shapes:
        plt.scatter(*np.array(shapes[s]).T, s=10, marker=(len(shapes[s]), 0, 0))

    plt.show()


def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--vertices', type=int, default=3, help="Number of shape vertices to start with")
    parser.add_argument(
        '-p', '--partitions', type=str, default="",
        help=(
            "Comma-separated list of how to split the vertices into shapes "
            "e.g. 3,6 splits N vertices into a shapes with 4 vertices, 6-3 vertices, and N-6 vertices"
        )
    )
    parser.add_argument('-t', '--iterations', type=int, default=1000, help="Timesteps to run the game for")
    args = parser.parse_args()

    vertices = args.vertices
    partitions = [int(p) for p in args.partitions.split(',') if p != ''] 

    # check params (ensure that shapes are at least triangles)
    vertex_partitions = [j - i for i, j in zip([0] + partitions, partitions + [vertices])] 
    assert all(p > 2 for p in vertex_partitions), (
        "Invalid partition. Please ensure each partition is at least 3 points"
    )

    # extract vertices on plot
    shapes = parse_points(vertices, vertex_partitions)

    # generate the random points
    points = chaos_game(shapes, iterations=args.iterations)

    plot_results(shapes, points)


if __name__ == '__main__':
    main()
