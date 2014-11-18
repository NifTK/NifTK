import restingstatefmri as fmri
import os

rs = fmri.RestingStatefMRIPreprocess()

data_path = '/Users/nicolastoussaint/data/nipype/fmri'

rs.inputs.in_t1 = os.path.join(data_path, "anat.nii.gz")
rs.inputs.in_fmri = os.path.join(data_path, "func.nii.gz")
rs.inputs.in_tissue_segmentation = os.path.join(data_path, "seg.nii.gz")
rs.inputs.in_parcellation = os.path.join(data_path, "atlas.nii.gz")

rs.run()
