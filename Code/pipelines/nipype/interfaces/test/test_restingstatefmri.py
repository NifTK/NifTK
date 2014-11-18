import restingstatefmri as fmri
import os

rs = fmri.RestingStatefMRIPreprocess()

data_path = '/Users/nicolastoussaint/data/nipype/fmri'

rs.inputs.in_subjectid = "test"
rs.inputs.in_fmri = os.path.join(data_path, "func.nii.gz")
rs.inputs.in_anat = os.path.join(data_path, "anat.nii.gz")
rs.inputs.in_group_avg = os.path.join(data_path, "avg.nii.gz")
rs.inputs.in_group_trans = os.path.join(data_path, "group_trans.nii.gz")
rs.inputs.in_seg = os.path.join(data_path, "seg.nii.gz")
rs.inputs.in_atlas = os.path.join(data_path, "atlas.nii.gz")

rs.run()
