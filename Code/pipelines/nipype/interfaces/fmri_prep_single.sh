#!/bin/bash
clear
echo The $0 command is called with $#argv parameters
echo SUBJ ID is $1
echo FMRI is $2
echo ANAT is $3
echo GROUP AVG is $4
echo GROUP TRANS is $5
echo SEG  is $6
echo ATLAS is $7


#echo All Parameters are \"$argv\"
#echo 2nd and on parameters are \"$argv[2-]\" 

bandpass_low=0.1
bandpass_high=0.01
smoothing_kernel_fwhm=3
tshift_pattern=alt+z
# =========================== auto block: setup ============================
# take note of the AFNI version
afni -ver

# set subject identifier
subj=$1
fmri=$2
anat=$3
group_avg=$4
group_trans=$5
seg=$6
atlas=$7

# assign output directory name
# echo ${fmri%%${fmri##*/}}
output_dir=$subj.results
# verify that the results directory does not yet exist
rm -R $output_dir
if [ -d $output_dir ]; then
    echo output dir "$subj.results" already exists
    exit
fi

# create results and stimuli directories
mkdir $output_dir
mkdir $output_dir/stimuli

# copy anatomy to results dir
#3dcopy $anat $output_dir/anat.nii.gz

# ============================ auto block: tcat ============================
# apply 3dTcat to copy input dsets to results dir, while
# removing the first 0 TRs
3dTcat -prefix $output_dir/pb00.$subj.r01.tcat ${fmri}'[0..$]'

# and make note of repetitions (TRs) per run
tr_counts=$( 3dinfo -nv $fmri  )
tr_time=$( 3dinfo -tr $fmri )
echo $tr_counts
echo $tr_time

# set list of runs
runs=$( count -digits 2 1 1 )

# -------------------------------------------------------
# enter the results directory (can begin processing data)
cd $output_dir

# ========================== auto block: outcount ==========================
# data check: compute outlier fraction for each volume
touch out.pre_ss_warn.txt
for run in $runs; do
  3dToutcount -automask -fraction -polort 3 -legendre pb00.$subj.r$run.tcat+orig > outcount.r$run.1D
    # censor outlier TRs per run, ignoring the first 0 TRs
    # - censor when more than 0.1 of automask voxels are outliers
    # - step() defines which TRs to remove via censoring
    1deval -a outcount.r$run.1D -expr "1-step(a-0.1)" > rm.out.cen.r$run.1D

    # outliers at TR 0 might suggest pre-steady state TRs
    if [ $( 1deval -a outcount.r$run.1D"{0}" -expr "step(a-0.4)" ) ]; then
        echo "** TR #0 outliers: possible pre-steady state TRs in run $run" \
            >> out.pre_ss_warn.txt
    fi
done

# catenate outlier counts into a single time series
cat outcount.r*.1D > outcount_rall.1D

# catenate outlier censor files into a single time series
cat rm.out.cen.r*.1D > outcount_${subj}_censor.1D

# get run number and TR index for minimum outlier volume
minindex=$( 3dTstat -argmin -prefix - outcount_rall.1D\' )
ovals=$( 1d_tool.py -set_run_lengths $tr_counts -index_to_run_tr $minindex )
# save run and TR indices for extraction of min_outlier_volume
#echo $ovals
IFS=', ' read -a array <<< "$ovals"
# for index in "${!array[@]}"
# do
#     echo "helo world"
#     echo "$index ${array[index]}"
# done
minoutrun=${array[0]}
minouttr=${array[1]}
echo "min outlier: run $minoutrun, TR $minouttr" | tee out.min_outlier.txt
# ================================ despike =================================
# apply 3dDespike to each run
for run in $runs; do
    3dDespike -NEW -nomask -prefix pb01.$subj.r$run.despike \
        pb00.$subj.r$run.tcat+orig
done

# ================================= tshift =================================
# time shift data so all slice timing is the same 
for run in $runs; do
    3dTshift -tzero 0 -quintic -prefix pb02.$subj.r$run.tshift \
             -tpattern ${tshift_pattern}                       \
             pb01.$subj.r$run.despike+orig
done

# copy min outlier volume as registration base
echo pb02.$subj.r$minoutrun.tshift+orig"[$minouttr]"
3dbucket -prefix min_outlier_volume                            \
    pb02.$subj.r$minoutrun.tshift+orig"[$minouttr]"

# ================================= volreg =================================
# align each dset to base volume
for run in $runs; do
    # register each volume to the base
    3dvolreg -verbose -zpad 1 -base min_outlier_volume+orig         \
             -1Dfile dfile.r$run.1D -prefix pb03.$subj.r$run.volreg \
             -cubic                                                 \
             pb02.$subj.r$run.tshift+orig

    # if there was an error, exit so user can see
    if [ $? -ne 0 ]; then 
      exit
    fi
done

# make a single file of registration params
cat dfile.r*.1D > dfile_rall.1D

# ================================== blur ==================================
# blur each volume of each run
for run in $runs; do
    3dmerge -1blur_fwhm ${smoothing_kernel_fwhm} -doall -prefix pb04.$subj.r$run.blur \
            pb03.$subj.r$run.volreg+orig
done

# ================================== mask ==================================
# create 'full_mask' dataset (union mask)
for run in $runs; do
    3dAutomask -dilate 1 -prefix rm.mask_r$run pb04.$subj.r$run.blur+orig
done

# create union of inputs, output type is byte
3dmask_tool -inputs rm.mask_r*+orig.HEAD -union -prefix full_mask.$subj

# ---- create subject anatomy mask, mask_anat.$subj+orig ----
#      (resampled from aligned anat)
3dresample -master full_mask.$subj+orig -input ${anat}     \
           -prefix rm.resam.anat

# convert to binary anat mask; fill gaps and holes
3dmask_tool -dilate_input 5 -5 -fill_holes -input rm.resam.anat+orig \
            -prefix mask_anat.$subj

# compute overlaps between anat and EPI masks
3dABoverlap -no_automask full_mask.$subj+orig mask_anat.$subj+orig   \
            | tee out.mask_ae_overlap.txt

# note correlation as well
3ddot full_mask.$subj+orig mask_anat.$subj+orig | tee out.mask_ae_corr.txt

# ================================== registration ==================================
3dcopy full_mask.$subj+orig full_mask.nii.gz
3dcopy min_outlier_volume+orig min_outlier_volume.nii.gz
reg_aladin -ref min_outlier_volume.nii.gz -flo ${anat} -rmask full_mask.nii.gz -aff anat_2_fmri_aff.txt -res anat_in_fmri_space.nii.gz -noSym   # use -rigOnly only when no scaling necessary for anat to fmri
reg_transform -invAff anat_2_fmri_aff.txt fmri_2_anat_aff.txt

#================================ regress =================================

# compute de-meaned motion parameters (for use in regression)
1d_tool.py -infile dfile_rall.1D -set_nruns 1                                \
           -demean -write motion_demean.1D

# compute motion parameter derivatives (for use in regression)
1d_tool.py -infile dfile_rall.1D -set_nruns 1                                \
           -derivative -demean -write motion_deriv.1D

# create censor file motion_${subj}_censor.1D, for censoring motion 
1d_tool.py -infile dfile_rall.1D -set_nruns 1                                \
    -show_censor_count -censor_prev_TR                                       \
    -censor_motion 0.3 motion_${subj}													

# combine multiple censor files
1deval -a motion_${subj}_censor.1D -b outcount_${subj}_censor.1D -expr "a*b" > censor_${subj}_combined_2.1D

# create bandpass regressors (instead of using 3dBandpass, say)
1dBport -nodata ${tr_counts} ${tr_time} -band ${bandpass_high} ${bandpass_low} -invert -nozero > bandpass_rall.1D                 ###########################


# ================================ regression model =================================
if [ ! -z "${seg}" -a "${seg}" != " " ] && [ ! -z "${atlas}" -a "${atlas}" != " " ]; then
	# ---- segment anatomy into classes CSF/GM/WM ----
  echo '---------------- YES SEGMENTATIONS ---------------------'
  3dcalc -a "${seg}[1]" -expr 'a' -prefix seg_csf.nii.gz
	3dcalc -a "${seg}[2]" -expr 'a' -prefix seg_gm.nii.gz
	3dcalc -a "${seg}[3]" -expr 'a' -prefix seg_wm.nii.gz
  3dcalc -a ${atlas} -expr 'within(a,52,53)' -prefix seg_ventricles.nii.gz
  reg_resample -ref min_outlier_volume.nii.gz -flo seg_csf.nii.gz -trans anat_2_fmri_aff.txt -res seg_csf_fmri.nii.gz -inter 0
  reg_resample -ref min_outlier_volume.nii.gz -flo seg_gm.nii.gz -trans anat_2_fmri_aff.txt -res seg_gm_fmri.nii.gz -inter 0
  reg_resample -ref min_outlier_volume.nii.gz -flo seg_wm.nii.gz -trans anat_2_fmri_aff.txt -res seg_wm_fmri.nii.gz -inter 0
  reg_resample -ref min_outlier_volume.nii.gz -flo seg_ventricles.nii.gz -trans anat_2_fmri_aff.txt -res seg_ventricles_fmri.nii.gz -inter 0

  3dcalc -a seg_csf_fmri.nii.gz -expr 'ispositive(a-0.8)' -prefix csf_mask.nii.gz
  3dcalc -a seg_gm_fmri.nii.gz -expr 'ispositive(a-0.5)' -prefix gm_mask.nii.gz
  3dcalc -a seg_wm_fmri.nii.gz -expr 'ispositive(a-0.8)' -prefix wm_mask.nii.gz
  3dcalc -a seg_ventricles_fmri.nii.gz -expr 'ispositive(a-0.6)' -prefix ventricles_mask.nii.gz 

  # create 2 ROI regressors: WMe, CSFe
  for run in $runs; do
    3dmaskave -quiet -mask wm_mask.nii.gz pb03.$subj.r$run.volreg+orig \
            | 1d_tool.py -infile - -demean -write rm.ROI.WMe.r$run.1D 
  	3dmaskave -quiet -mask ventricles_mask.nii.gz pb03.$subj.r$run.volreg+orig \
            | 1d_tool.py -infile - -demean -write rm.ROI.CSFe.r$run.1D 
  done
    
  # and catenate the demeaned ROI averages across runs
	cat rm.ROI.WMe.r*.1D > ROI.WMe_rall.1D
	cat rm.ROI.CSFe.r*.1D > ROI.CSFe_rall.1D

	# run the regression analysis
	3dDeconvolve -input pb04.$subj.r*.blur+orig.HEAD                             \
	    -censor censor_${subj}_combined_2.1D                                     \
	    -ortvec bandpass_rall.1D bandpass                                        \
	    -ortvec ROI.WMe_rall.1D ROI.WMe                                          \
	    -ortvec ROI.CSFe_rall.1D ROI.CSFe                                        \
	    -polort 3                                                                \
	    -num_stimts 12                                                           \
	    -stim_file 1 motion_demean.1D'[0]' -stim_base 1 -stim_label 1 roll_01    \
	    -stim_file 2 motion_demean.1D'[1]' -stim_base 2 -stim_label 2 pitch_01   \
	    -stim_file 3 motion_demean.1D'[2]' -stim_base 3 -stim_label 3 yaw_01     \
	    -stim_file 4 motion_demean.1D'[3]' -stim_base 4 -stim_label 4 dS_01      \
	    -stim_file 5 motion_demean.1D'[4]' -stim_base 5 -stim_label 5 dL_01      \
	    -stim_file 6 motion_demean.1D'[5]' -stim_base 6 -stim_label 6 dP_01      \
	    -stim_file 7 motion_deriv.1D'[0]' -stim_base 7 -stim_label 7 roll_02     \
	    -stim_file 8 motion_deriv.1D'[1]' -stim_base 8 -stim_label 8 pitch_02    \
	    -stim_file 9 motion_deriv.1D'[2]' -stim_base 9 -stim_label 9 yaw_02      \
	    -stim_file 10 motion_deriv.1D'[3]' -stim_base 10 -stim_label 10 dS_02    \
	    -stim_file 11 motion_deriv.1D'[4]' -stim_base 11 -stim_label 11 dL_02    \
	    -stim_file 12 motion_deriv.1D'[5]' -stim_base 12 -stim_label 12 dP_02    \
	    -fout -tout -x1D X.xmat.1D -xjpeg X.jpg                                  \
	    -x1D_uncensored X.nocensor.xmat.1D                                       \
	    -fitts fitts.$subj                                                       \
	    -errts errts.${subj}                                                     \
	    -bucket stats.$subj

########################### alternatively do not use tissue regressors #########################
else
  echo '---------------- NO SEGMENTATIONS ---------------------'
	# run the regression analysis
  3dDeconvolve -input pb04.$subj.r*.blur+orig.HEAD                             \
      -censor censor_${subj}_combined_2.1D                                     \
      -ortvec bandpass_rall.1D bandpass                                        \
      -polort 3                                                                \
      -num_stimts 12                                                           \
      -stim_file 1 motion_demean.1D'[0]' -stim_base 1 -stim_label 1 roll_01    \
      -stim_file 2 motion_demean.1D'[1]' -stim_base 2 -stim_label 2 pitch_01   \
      -stim_file 3 motion_demean.1D'[2]' -stim_base 3 -stim_label 3 yaw_01     \
      -stim_file 4 motion_demean.1D'[3]' -stim_base 4 -stim_label 4 dS_01      \
      -stim_file 5 motion_demean.1D'[4]' -stim_base 5 -stim_label 5 dL_01      \
      -stim_file 6 motion_demean.1D'[5]' -stim_base 6 -stim_label 6 dP_01      \
      -stim_file 7 motion_deriv.1D'[0]' -stim_base 7 -stim_label 7 roll_02     \
      -stim_file 8 motion_deriv.1D'[1]' -stim_base 8 -stim_label 8 pitch_02    \
      -stim_file 9 motion_deriv.1D'[2]' -stim_base 9 -stim_label 9 yaw_02      \
      -stim_file 10 motion_deriv.1D'[3]' -stim_base 10 -stim_label 10 dS_02    \
      -stim_file 11 motion_deriv.1D'[4]' -stim_base 11 -stim_label 11 dL_02    \
      -stim_file 12 motion_deriv.1D'[5]' -stim_base 12 -stim_label 12 dP_02    \
      -fout -tout -x1D X.xmat.1D -xjpeg X.jpg                                  \
      -x1D_uncensored X.nocensor.xmat.1D                                       \
      -fitts fitts.$subj                                                       \
      -errts errts.${subj}                                                     \
      -bucket stats.$subj
  exit
fi

# if 3dDeconvolve fails, terminate the script
# question status
if [ $? -ne 0 ]; then
    echo '---------------------------------------'
    echo '** 3dDeconvolve error, failing...'
    echo '   (consider the file 3dDeconvolve.err)'
    exit
else
    echo '---------------------------------------'
    echo '** 3dDeconvolve worked fine...'
fi


# display any large pariwise correlations from the X-matrix
1d_tool.py -show_cormat_warnings -infile X.xmat.1D | tee out.cormat_warn.txt

# create an all_runs dataset to match the fitts, errts, etc.
3dTcat -prefix all_runs.$subj pb04.$subj.r*.blur+orig.HEAD

# --------------------------------------------------
# create a temporal signal to noise ratio dataset 
#    signal: if 'scale' block, mean should be 100
#    noise : compute standard deviation of errts
3dTstat -mean -prefix rm.signal.all all_runs.$subj+orig
3dTstat -stdev -prefix rm.noise.all errts.${subj}+orig
3dcalc -a rm.signal.all+orig                                                 \
       -b rm.noise.all+orig                                                  \
       -c full_mask.$subj+orig                                               \
       -expr 'c*a/b' -prefix TSNR.$subj 

# ---------------------------------------------------
# compute and store GCOR (global correlation average)
# (sum of squares of global mean of unit errts)
3dTnorm -norm2 -prefix rm.errts.unit errts.${subj}+orig
3dmaskave -quiet -mask full_mask.$subj+orig rm.errts.unit+orig >             \
    gmean.errts.unit.1D
3dTstat -sos -prefix - gmean.errts.unit.1D\' > out.gcor.1D
echo "-- GCOR = $( cat out.gcor.1D )"

# --------------------------------------------------------
# compute sum of non-baseline regressors from the X-matrix
# (use 1d_tool.py to get list of regressor colums)
reg_cols=$( 1d_tool.py -infile X.nocensor.xmat.1D -show_indices_interest )
3dTstat -sum -prefix sum_ideal.1D X.nocensor.xmat.1D"[$reg_cols]"

# also, create a stimulus-only X-matrix, for easy review
1dcat X.nocensor.xmat.1D"[$reg_cols]" > X.stim.xmat.1D

# ============================ blur estimation =============================
# compute blur estimates
touch blur_est.$subj.1D   # start with empty file

# -- estimate blur for each run in errts --
touch blur.errts.1D

# restrict to uncensored TRs, per run
for run in $runs; do 
    trs=$( 1d_tool.py -infile X.xmat.1D -show_trs_uncensored encoded     \
                          -show_trs_run $run )
    if [ $trs == "" ]; then 
      continue
    fi
    3dFWHMx -detrend -mask full_mask.$subj+orig                              \
        errts.${subj}+orig"[$trs]" >> blur.errts.1D
done

# compute average blur and append
blurs=$( cat blur.errts.1D )
echo average errts blurs: $blurs
echo "$blurs   # errts blur estimates" >> blur_est.$subj.1D


# ================== auto block: generate review scripts ===================

# generate a review script for the unprocessed EPI data
# gen_epi_review.py -script @epi_review.$subj \
#     -dsets pb00.$subj.r*.tcat+orig.HEAD

# # generate scripts to review single subject results
# # (try with defaults, but do not allow bad exit status)
# gen_ss_review_scripts.py -mot_limit 0.3 -out_limit 0.1 -exit0

# # ========================== auto block: finalize ==========================

# # if the basic subject review script is here, run it
# # (want this to be the last text output)
# if [-e @ss_review_basic ]; then 
#   ./@ss_review_basic | tee out.ss_review.$subj.txt
# fi

## =========================== create final copy of preprocessed fmri scan ======================================
3dcopy errts.${subj}+orig ../${subj}.fmri_pp.nii.gz
if [ ! -z "${group_avg}" -a "${group_avg}" != " " ] && [ ! -z "${group_trans}" -a "${group_trans}" != " " ]; then
  reg_transform -comp fmri_2_anat_aff.txt $group_trans fmri_2_group.nii.gz -ref ${group_avg}
  reg_resample -ref ${group_avg} -flo ../${subj}.fmri_pp.nii.gz -trans fmri_2_group.nii.gz -res ../${subj}.fmri_pp_group.nii.gz
  reg_resample -ref ${group_avg} -flo min_outlier_volume.nii.gz -trans fmri_2_group.nii.gz -res ../${subj}.fmri_reg_group.nii.gz
fi

# return to parent directory
cd ..

echo "execution finished: `date`"