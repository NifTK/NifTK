from nipype.interfaces.afni.proc import RestingStatefMRIPreprocess as RS
rs = RS()
rs.inputs.in_subjectid = "test"
rs.inputs.in_fmri = "/Users/me/data/pipeline_test/func.nii.gz"
rs.inputs.in_anat = "/Users/me/data/pipeline_test/anat.nii.gz"
rs.inputs.in_group_avg = "/Users/me/data/pipeline_test/avg.nii.gz"
rs.inputs.in_group_trans = "/Users/me/data/pipeline_test/group_trans.nii.gz"
rs.inputs.in_seg = "/Users/me/data/pipeline_test/seg.nii.gz"
rs.inputs.in_atlas = "/Users/me/data/pipeline_test/atlas.nii.gz"
rs.run()
