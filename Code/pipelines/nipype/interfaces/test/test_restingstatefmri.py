import restingstatefmri as fmri

rs = fmri.RestingStatefMRIPreprocess()

rs.inputs.in_subjectid = "test"
rs.inputs.in_fmri = "func.nii.gz"
rs.inputs.in_anat = "anat.nii.gz"
rs.inputs.in_group_avg = "avg.nii.gz"
rs.inputs.in_group_trans = "group_trans.nii.gz"
rs.inputs.in_seg = "seg.nii.gz"
rs.inputs.in_atlas = "atlas.nii.gz"

rs.run()
