
current_dir=`dirname $0`
data_dir=${current_dir}/../../../Testing/Data/Input
ouput_dir=/var/NOT_BACKUP/results/groupwise-ffd-test

#./ComputeInitialAtlas.sh ${ouput_dir}/ellipse_atlas.hdr \
#                         ${ouput_dir}/ellipse_atlas_init_%i \
#                         ${data_dir}/ellipse-128-128-128-50-45-40.hdr \
#                         ${data_dir}/ellipse-128-128-128-50-45-40.hdr \
#                         0 \
#                         ${data_dir}/ellipse-128-128-128-50-50-50.hdr

#./ComputeGroupwiseAtlas.sh ${ouput_dir}/ellipse_atlas_nreg_%i \
#                           ${ouput_dir}/ellipse_nreg_%i \
#                           ${ouput_dir}/ellipse_diff.txt \
#                           ${data_dir}/ellipse-128-128-128-50-45-40.hdr \
#                           ${data_dir}/ellipse-128-128-128-50-45-40.hdr \
#                           ${data_dir}/ellipse-128-128-128-50-45-40.hdr \
#                           ${ouput_dir}/ellipse_atlas_init_0-nreg-init.dof \
#                           ${data_dir}/ellipse-128-128-128-50-50-50.hdr \
#                           ${ouput_dir}/ellipse_atlas_init_1-nreg-init.dof 
                           
                           
                           
                           
#./ComputeInitialAtlas.sh ${ouput_dir}/ellipse_atlas_backward.hdr \
#                         ${ouput_dir}/ellipse_atlas_backward_init_%i \
#                         ${data_dir}/ellipse-128-128-128-50-50-50.hdr \
#                         ${data_dir}/ellipse-128-128-128-50-50-50.hdr \
#                         0 \
#                         ${data_dir}/ellipse-128-128-128-50-45-40.hdr \
                         

./ComputeGroupwiseAtlas.sh ${ouput_dir}/ellipse_atlas_backward_nreg_%i \
                           ${ouput_dir}/ellipse_nreg_backward_%i \
                           ${ouput_dir}/ellipse_backward_diff.txt \
                           ${data_dir}/ellipse-128-128-128-50-50-50.hdr \
                           ${data_dir}/ellipse-128-128-128-50-50-50.hdr \
                           ${data_dir}/ellipse-128-128-128-50-50-50.hdr \
                           ${ouput_dir}/ellipse_atlas_backward_init_0-nreg-init.dof \
                           ${data_dir}/ellipse-128-128-128-50-45-40.hdr \
                           ${ouput_dir}/ellipse_atlas_backward_init_1-nreg-init.dof 
