#/*============================================================================
#
#  NifTK: A software platform for medical image computing.
#
#  Copyright (c) University College London (UCL). All rights reserved.
#
#  This software is distributed WITHOUT ANY WARRANTY; without even
#  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
#  PURPOSE.
#
#  See LICENSE.txt in the top level directory for details.
#
#============================================================================*/

MITK_CREATE_MODULE_TESTS(LABELS niftkCaffe)

if(TARGET ${TESTDRIVER})

  mitk_use_modules(TARGET ${TESTDRIVER} PACKAGES Caffe)

  add_test(Caffe-Liver1 ${CXX_TEST_PATH}/niftkCaffeTestDriver
           niftkCaffeSegmentImageTest
           ${NIFTK_DATA_DIR}/Input/Caffe/Liver-2016-09-27/resnet3.deploy.with-memory-layer.prototxt
           ${NIFTK_DATA_DIR}/Input/Caffe/Liver-2016-09-27/resnet3.leaveOutLS08_iter10000.caffemodel
           prediction                     # name of output node
           321 81                         # size (x, y) of expected input
            81 21                         # size (x, y) of expected output
           -128.65884,-66.821,-58.987     # additive offset for input pixels (RGB): (-128,-128,-128) - (0.65884,-61.179,-69.013)
           ${NIFTK_DATA_DIR}/Input/Caffe/Liver-2016-09-27/1449229588798812400_left.png
           ${NIFTK_DATA_DIR}/Input/Caffe/Liver-2016-09-27/1449229588798812400_leftMask.png
           ${TEMP}/Caffe-Liver1-output.png
           )

  add_test(Caffe-Phantom1 ${CXX_TEST_PATH}/niftkCaffeTestDriver
           niftkCaffeSegmentImageTest
           ${NIFTK_DATA_DIR}/Input/Caffe/Phantom-2016-09-27/phantom.deploy.with-memory-layer.prototxt
           ${NIFTK_DATA_DIR}/Input/Caffe/Phantom-2016-09-27/snapshot.resnet3.phantom_iter_300.caffemodel
           prediction                     # name of output node
           321 81                         # size (x, y) of expected input
            81 21                         # size (x, y) of expected output
           -128,-128,-128                 # additive offset for input pixels
           ${NIFTK_DATA_DIR}/Input/Caffe/Phantom-2016-09-27/1386775970515321200_left.png
           ${NIFTK_DATA_DIR}/Input/Caffe/Phantom-2016-09-27/1386775970515321200_leftMask.png
           ${TEMP}/Caffe-Phantom1-output.png
           )
endif()

