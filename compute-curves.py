import pandas as pd
import numpy as np
import gtda

import gtda.diagrams as diag
import sklearn.preprocessing as skprep
from gtda.homology import CubicalPersistence, WeakAlphaPersistence
from gtda.images import DensityFiltration
from gtda.pipeline import Pipeline

steps = [
    ('rescaler', diag.Scaler()),
    ('filter', diag.Filtering(epsilon=0.01)),
    ('feature', diag.BettiCurve(n_bins=100)),
]
pipeline = Pipeline(steps)

filtration = DensityFiltration(n_jobs=10)

pers = CubicalPersistence(n_jobs=10, homology_dimensions=[0, 1, 2])

print('Computing filtration on gray matter images...')
gm = np.load('Data/gray-matter.npy')
gm = filtration.fit_transform(gm)
print('Computing gray matter homology...')
diags = cube.fit_transform(gm)
print('Saving...')
np.save('Data/gm-diags.npy', diags)
gm_curves = pipeline.fit_transform(diags)
np.save('Data/gm-curves.npy', gm_curves)
del gm

print('Computing white matter homology...')
wm = np.load('Data/white-matter.npy')
diags = cube.fit_transform(wm)
np.save('Data/wm-diags.npy', diags)
wm_curves = pipeline.fit_transform(diags)
print('Saving...')
np.save('Data/wm-curves.npy', wm_curves)
del wm

pers = WeakAlphaPersistence(n_jobs=10, homology_dimensions=[0, 1])

print('Computing point cloud homology...')
pcl = np.load('Data/point-cloud.npy')
diags = pers.fit_transform(pcl)
pcl_curves = pipeline.fit_transform(diags)
print('Saving...')
np.save('Data/pcl-diags.npy', diags)
np.save('Data/pcl-curves.npy', pcl_curves)
del pcl

print('Saving combined betti curves...')
curves = np.concatenate([gm_curves, wm_curves, pcl_curves], axis=1)
np.save('Data/full-curves.npy')
