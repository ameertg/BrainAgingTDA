import pandas as pd
import os
from tqdm import tqdm
from nilearn.surface import load_surf_data
import numpy as np
from nibabel.freesurfer.io import read_geometry
import pickle as p


# Change to name of directory containing freesurfer data
data_dir = './oasis'

dat = {}

for root, dirs, files in os.walk(data_dir):
    for file in files:
        if file == 'brain.mgz':
            subj = root[11:-4]
            if subj not in dat:
                dat[subj] = {'brain': os.path.join(root, file)}
            else:
                dat[subj]['brain'] = os.path.join(root, file)
        if file == 'wm.mgz':
            subj = root[11:-4]
            if subj not in dat:
                dat[subj] = {'wm': os.path.join(root, file)}
            else:
                dat[subj]['wm'] = os.path.join(root, file)
        if file == 'aseg.mgz':
            subj = root[11:-4]
            if subj not in dat:
                dat[subj] = {'aseg': os.path.join(root, file)}
            else:
                dat[subj]['aseg'] = os.path.join(root, file)
        if file == 'lh.thickness':
            subj = root[11:-5]
            if subj not in dat:
                dat[subj] = {'lhthick': os.path.join(root, file)}
            else:
                dat[subj]['lhthick'] = os.path.join(root, file)
        if file == 'rh.thickness':
            subj = root[11:-5]
            if subj not in dat:
                dat[subj] = {'rhthick': os.path.join(root, file)}
            else:
                dat[subj]['rhthick'] = os.path.join(root, file)
        if file == 'lh.area':
            subj = root[11:-5]
            if subj not in dat:
                dat[subj] = {'lharea': os.path.join(root, file)}
            else:
                dat[subj]['lharea'] = os.path.join(root, file)
        if file == 'rh.area':
            subj = root[11:-5]
            if subj not in dat:
                dat[subj] = {'rharea': os.path.join(root, file)}
            else:
                dat[subj]['rharea'] = os.path.join(root, file)

print('Computing cortical thickness... WARNING DO NOT STOP THIS PROCESS BEFORE IT IS COMPLETED')
for key in tqdm(dat.keys()):
    lhfile =  dat[key]['lharea']
    os.rename(lhfile, lhfile + '.thickness')
    lhfile = lhfile + '.thickness'
    rhfile =  dat[key]['rharea']
    os.rename(rhfile, rhfile + '.thickness')
    rhfile = rhfile + '.thickness'


    lhthickfile = dat[key]['lhthick']
    rhthickfile = dat[key]['rhthick']
    files = [lhfile, rhfile, lhthickfile, rhthickfile]

    lharea, rharea, lhthick, rhthick = map(lambda x: load_surf_data(x), files)

    dat[key]['avgthickness'] = (np.sum(lhthick * lharea) + np.sum(rhthick * rharea)) / (np.sum(rharea) + np.sum(lharea))

    os.rename(rhfile, dat[key]['rharea'])
    os.rename(lhfile, dat[key]['lharea'])

df = pd.DataFrame.from_dict(dat, orient='index').reset_index()
df = df.rename({'index': 'MR ID'}, axis=1)
df['MR ID'] = 'OAS' + df['MR ID']
mrs = pd.read_csv('Data/Oasis_MRI.csv')
clinical = pd.read_csv('Data/Oasis_clinicalData.csv')
freesurfer = pd.read_csv('Data/Oasis_FS.csv')

freesurfer['MR ID'] = freesurfer['FS_FSDATA ID'].str.replace('_Freesurfer[0-9][0-9]_', '_MR_', regex=True)

df = pd.merge(df, mrs[['MR ID', 'Age']], how='inner', on='MR ID')
df = pd.merge(df, freesurfer.drop(['FS_FSDATA ID', 'Session', 'FS Date', 'Included T1s'], axis=1), how='inner', on='MR ID')

sick = np.unique(clinical[clinical['cdr'] > 0]['Subject'])
df['isSick'] = df['Subject'].isin(sick)
df = df[~df['Age'].isna()].reset_index(drop=True)
df.to_csv('Data/subject_data.csv')


grays = []
whites = []
errors = []
print('Loading images...')
for n, row in tqdm(df.iterrows(), total=df.shape[0]):
    try:
        brain = load_surf_data(row['brain'])
        segmentation = load_surf_data(row['aseg'])
        gm = brain * np.logical_not(np.logical_or(segmentation==41, segmentation==2))
        gm = gm * np.logical_not(np.logical_or(segmentation==46, segmentation==7))
        grays.append(gm)
        wm = load_surf_data(row['wm']) > 0
        whites.append(brain * wm)
    except EOFError as error:
        print('Failed to load {row['MR ID']}')
        print('This subject will be removed from the dataset.')
        errors.append('MR ID')

df = df[~df['MR ID'].isin(errors)]
df.to_csv('Data/subject_data.csv')
print('Final size of dataset: {df.shape[0]}')
whites = np.array(whites)
grays = np.array(grays)

print('Saving images...')
np.save('Data/white-matter.npy', whites)
np.save('Data/gray-matter.npy', grays)

print('Loading point clouds...')
pcls = []
for image in tqdm(df['lharea']):
    image = image[:-7]
    lh = read_geometry(image + 'lh.pial')[0]
    rh = read_geometry(image + 'rh.pial')[0]
    pcls.append(np.concatenate([lh, rh], axis=0))

print('Saving point clouds...')
np.save('Data/point-cloud.npy', pcls)
