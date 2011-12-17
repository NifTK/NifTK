########################
# BSI SOFTWARE PACKAGE #
########################

##############################################################################

--------------------------------
1 WHAT DOES THE PACKAGE CONTAIN?
--------------------------------
The code contains a program to calculate the boundary shift integral to 
measure the brain atrophy over the baseline and registered follow-up 
3D images using an improved intensity normaliation and automatic intensity 
window selection method. 

Freeborough and Fox [1] presented an algorithm called the boundary shift
integral (BSI) to measure the brain atrophy using longitudinal MRI scans. 
Leung et al. [2] described an improved intensity normalization and automatic
intensity window selection method for calculating BSI using MRI scans from 
multiple imaging sites. This improved BSI is referred to as KN-BSI. 

The code has been implemented using C/C++ language and the Insight Toolkit. 

If you are planning to use any of our research, we would be grateful if you
would be kind enough to cite references [1] and [2]. 

##############################################################################

-----------------------
2 HOW TO BUILD THE CODE 
-----------------------
The code can be easily build using cmake (http://www.cmake.org/). The latest 
version can be downloaded from http://www.cmake.org/cmake/resources/software.html
Assuming that the code source are in the source path folder, you will have 
to ﬁrst create a new folder, i.e. build path (#1) and then to change 
directory to move into that folder (#2).
#1 >> mkdir build path 
#2 >> cd build path 

There you will need to call ccmake (#3a) in order to fill in the 
build options. If you don’t want to specify options, we could just use cmake 
(#3b) and the default build values will be used.
#3a >> ccmake source path
#3b >> cmake source path

The main options in the ccmake are deﬁned bellow:
>CMAKE_BUILD_INSTALL options are Release, RelWithDebInfo or Debug
>CMAKE_INSTALL_PREFIX installation path
>ITK_DIR Insight Toolkit library path

Once all the ﬂags are properly ﬁlled in, just press the ”c” to conﬁgure the Make- 
ﬁle and then the ”g” key to generate them. In the prompt, you just have to 
make (#4) ﬁrst and then make install (#5).
#4 >> make 
#5 >> make install 

##############################################################################

---------
3 EXAMPLE
---------
In this example, we will calculate the KN-BSI from the baseline and registered
follow-up images images called basleine.hdr and repeat.hdr, and the baseline 
and follow-up brain regions called baseline_mask.hdr and repeat_mask.hdr. 

# niftkKN-BSI baseline.hdr baseline_mask.hdr \
              repeat.hdr repeat_mask.hdr \
              baseline.hdr baseline_mask.hdr \
              repeat.hdr repeat_mask.hdr \              
              1 1 3 -1 -1  \
              baseline_seg.hdr repeat_seg.hdr repeat_norm.hdr 

The last number in the output is the value of KN-BSI.

The baseline_seg.hdr and repeat_seg.hdr are the output segmentation using the
k-means clustering algorithm in Insight Toolkit. The repeat_norm.hdr is the
normalized repeat image. 

##############################################################################

---------
4 LICENSE
---------
See SoftwareLicence.txt. 

##############################################################################

---------
5 CONTACT
---------
For any comment, please, feel free to contact Kelvin Leung (kk.leung@ucl.ac.uk).

##############################################################################

------------
6 REFERENCES
------------
[1] Freeborough PA and Fox NC, The boundary shift integral: an accurate and 
robust measure of cerebral volume changes from registered repeat MRI IEEE 
Trans Med Imaging. 1997 Oct;16(5):623-9

[2] Leung KK, Clarkson MJ, Bartlett JW, Clegg S, Jack CR Jr, Weiner MW, Fox NC, 
Ourselin S; the Alzheimer's Disease Neuroimaging Initiative. Robust atrophy 
rate measurement in Alzheimer's disease using multi-site serial MRI: 
Tissue-specific intensity normalization and parameter selection. 
Neuroimage. 2009 Dec 23. 

##############################################################################
##############################################################################
##############################################################################
##############################################################################

