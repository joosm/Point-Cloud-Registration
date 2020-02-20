import numpy as np

test = np.load('transformations.npz')
print(test['initial_transformation'])
print(test['refined_transformation'])