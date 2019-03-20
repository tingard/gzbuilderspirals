# Galaxy Builder Spiral Aggregation

This package is intended to be a convenience package allowing the aggregation of many poly-lines into separate logarithmic spirals. It was built for the Galaxy Builder citizen science project on the [Zooniverse](https://www.zooniverse.org) web platform.

The reccomended use is via `gzbuilderspirals.oo.Pipeline` as follows

~~~python
p = Pipeline(poly_lines, phi=GALAXY_ROTATION, ba=GALAXY_AXIS_RATIO,
             image_size=ZOONIVERSE_IMAGE_SIZE)
arms = p.get_arms()
pitch_angle, pitch_angle_error = p.get_pitch_angle(arms)
~~~

Where `GALAXY_ROTATION` and `GALAXY_AXIS_RATIO` correspond to the elliptical properties of the galaxy isophote (for instance **PETRO_PHI90** and **SERSIC_BA** from the [NASA-Sloan Atlas](https://www.sdss.org/dr13/manga/manga-target-selection/nsa/)), and `ZOONIVERSE_IMAGE_SIZE` is the square image size seen by volunteers.

`Pipeline` has many optional keyword-arguments:

- `phi`, `ba` describe galaxy position angle and axis ratio, which measure its inclination
- `distances`, a pre-calculated array of distances between drawn arms which can be used instead of the custom metric used
- `bar_length` length of bar, inside which all points are ignored during the fitting process
- `centre_size` threshold size inside which points are deleted. Arms which go in and out of the centre are split
- `image_size` the square image size seen by volunteers, and is used to identify the centre of the image in order to transform from caretsian to polar coordinates for sorting and fitting
- `clustering_kwargs` keyword-arguments to pass through to the internal [DBSCAN](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html) clustering algorithm
- `parallel` whether to use a `multiprocessing.Pool` when calculating the distance matrix (causes errors in some environments)

### Logarithmic Spiral Extraction Process

*Inside the `gzbuilderspirals.oo.Pipeline` class:*

- Arms are split inside the `centre_size`
- If no distance matrix is provided, the custom metric described in [Lingard (in prep)](/#) is used to create one.
- Distances are fed into the DBSCAN clustering algorithm and poly-lines are grouped according to their cluster

*Inside the `gzbuilderspirals.oo.Arm` class:*

- Inside each cluster, poly-lines are interpolated such that they all contain the same number of points, equal to the maximum number of points in any of the cluster member poly-lines
- Points are unwrapped such that theta is no longer clamped between 0 and 360, and instead is monotonic(ish) with distance along poly-line
	-  This step occasionally fails for clusters containing poly-lines which behave erratically
- Points are cleaned using the [Local Outlier Factor](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.LocalOutlierFactor.html#sklearn.neighbors.LocalOutlierFactor) unsupervised outlier detection algorithm, which measures the local deviation of density of a given sample with respect to its neighbours. This respects the grouped nature of points by testing each group against the model trained on all other groups.
- Point weights are calculated based on radius from centre (small radius = low weight) and by the number of other points also present at a given radius (more points at a given radius = higher weight)
- Bayesian Ridge Regression is used to fit the log spiral model. Hyper-parameters were obtained by fitting a truncated gamma distribution to the widths of spiral arms (obtained from "spread" sliders presented to volunteers). This is tailored to square images where 512 pixels corresponds to 4x the galaxy's petrosean radius.
- The logarithmic spiral's pitch angle is recovered from the fitted model parameters

Logarithmic spiral:
$$r = ae^{\phi\tan(\theta)}$$

Where $r$ and $\phi$ are the polar coordinates and $\theta$ is the pitch angle of the spiral (angle between the curve and a circle of radius $r$).
