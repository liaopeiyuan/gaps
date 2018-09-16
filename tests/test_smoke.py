import pytest
import numpy as np
import cv2

from gaps import image_helpers
from gaps.genetic_algorithm import GeneticAlgorithm
from PIL import Image
from keras.preprocessing.image import load_img


GENERATIONS = 20
POPULATION = 300
PIECE_SIZE = 101

image = np.array(cv2.imread("../train.png"))
print(image.shape)

@pytest.fixture
def puzzle():
    pieces, rows, columns = image_helpers.flatten_image(image, PIECE_SIZE)
    np.random.shuffle(pieces)
    return image_helpers.assemble_image(pieces, rows, columns)

def test_puzzle_solver(puzzle):
    algorithm = GeneticAlgorithm(puzzle, PIECE_SIZE, POPULATION, GENERATIONS)
    solution = algorithm.start_evolution(verbose=False)
    A=solution.to_image()
    print(type(A))
    im = Image.fromarray(A)
    im.show()
    assert np.array_equal(image, solution.to_image())

test_puzzle_solver(image)