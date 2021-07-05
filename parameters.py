# Resize fraction of images 
# Lower values will result in higher quality images, but also longer computation time
# Different values possible require adjustments of threshold parameters for optimal results
# 1 will not resize the image
# Default: 4
resize = 4

# Minimum threshold for symmetry scores of main reflection symmetry lines
# Lines with scores below this value will be removed, along with all lines discovered in the same recursive loop.
# This threshold only dictates which lines will be fed to the rotational symmetry machine learning model
# Has no function in 'fast' mode
# Default: 0.09
symThresholdBC = 0.09

# Minimum threshold for normalized symmetry scores of reflection symmetry lines.
# Lines with normalized scores below this value will be removed.
# This threshold only dictates which lines will be fed to the rotational symmetry machine learning model
# Has no function in 'fast' mode
# Default: 0.375
normThresholdBC = 0.375

# Minimum threshold for symmetry scores of main reflection symmetry lines
# Lines with scores below this value will be removed, along with all lines discovered in the same recursive loop.
# Default: 0.25
symThresholdAC = 0.25

# Minimum threshold for normalized symmetry scores of reflection symmetry lines.
# Lines with normalized scores below this value will be removed.
# Default: 0.75
normThresholdAC = 0.75

# Maximum relative score difference between two lines in order for them to form a rotational symmetry in 'fast' mode.
# Has no function in 'slow' mode.
# Default: 0.75
circleSymThreshold = 0.75

# Dictates maximum distance between line endpoint for them to be flagged as similar and subsequently be removed
# Maximum distances are calculated as follows: (height or width) / lineSimilarity
# Lower values cause lines to be sooner flagged as similar
# default = 8
lineSimilarity = 8 #Lower values cause lines/rotations to be flagged as similar and removed sooner.

# Dictates maximum distance between centerpoints and radii of rotational symmetries for them to be flagged as similar and subsequently be removed
# Maximum distances are calculated as follows: (height or width) / rotationSimilarity
# Lower values cause rotational symmetries to be sooner flagged as similar
# default = 3
rotationSimilarity = 3 #Lower values cause lines/rotations to be flagged as similar and removed sooner.

