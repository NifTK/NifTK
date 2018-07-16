Welcome
-------

This is the @NIFTK_PLATFORM@ Translational Software Platform:
Version @NIFTK_VERSION_STRING@.

For all questions and queries, please join the users mailing list:
@NIFTK_USER_CONTACT@.

Notices
-------

We have included the Slicer Execution Model. This is downloaded at compile
time, in order to build command line applications. In addition the file
NifTK/CMake/niftkMacroBuildCLI.cmake contains a modified version of 
SEMMacroBuildCLI.cmake. The Slicer license requires the following notice:

"All or portions of this licensed product (such portions are the "Software") 
have been obtained under license from The Brigham and Women's Hospital, Inc. 
and are subject to the following terms and conditions:
  http://www.slicer.org/copyright/copyright.txt"

Additional Libraries
--------------------

This platform uses these libraries/projects listed below. 
Please check the NifTK/Documentation/Licenses directory for licenses.
These licenses should also be installed with the packaged binaries.
The exact number of packages installed depends on configure 
parameters at compile time.
     
Library : Boost
Website : http://www.boost.org
Purpose : Generic library of high quality C++ code. 
License : Documentation/Licenses/Boost.txt (Boost Software License)

Library : VTK
Website : http://www.vtk.org
Purpose : Graphics and visualization.
License : Documentation/Licenses/VTK.txt (BSD 3 clause license)

Library : GDCM
Website : http://www.creatis.insa-lyon.fr/software/public/Gdcm
Purpose : DICOM file loading library.
License : Documentation/Licenses/GDCM.txt (BSD 3 clause license)

Library : DCMTK
Website : http://dicom.offis.de
Purpose : DICOM file loading and networking.
License : Documentation/Licenses/DCMTK.txt (mainly BSD 3 clause, but a mixture, please read this)

Library : ITK
Website : http://www.itk.org
Purpose : Image processing algorithms.
License : Documentation/Licenses/ITK.txt (Apache v2 license)
Version : 4.7.1 Release
Mods    : Applied via CMake patches:
          https://cmiclab.cs.ucl.ac.uk/CMIC/NifTK/tree/master/CMake/ExternalProjects

Library : OpenCV
Website : http://opencv.willowgarage.com/wiki
Purpose : Computer Vision.
License : Doc/Licences/OpenCV.txt (BSD 3 clause license)

Library : ArUco
Website : http://www.uco.es/investiga/grupos/ava/node/26
Purpose : AR marker tracking.
License : Documentation/Licenses/ArUco.txt (BSD 2 clause license)

Library : Eigen
Website : http://eigen.tuxfamily.org
Purpose : Matrix library used by AprilTags, PCL etc.
License : http://www.mozilla.org/MPL/2.0/ (Mozilla Public License v2.0)

Library : AprilTags
Website : http://people.csail.mit.edu/kaess/apriltags
Purpose : AR marker tracking.
License : Documentation/Licenses/AprilTags.txt (LGPL v2.1 license)

Library : FLANN
Website : http://www.cs.ubc.ca/research/flann
Purpose : Needed by PCL, fast, approx nearest neighbour algorithm.
License : Documentation/Licenses/FLANN.txt (BSD 3 clause license)

Library : PCL
Website : http://pointclouds.org 
Purpose : Point cloud processing.
License : Documentation/Licenses/PCL.txt (BSD 3 clause license)

Library : MITK
Website : http://www.mitk.org
Purpose : Application framework, common medical imaging specific classes.
License : Documentation/Licenses/MITK.txt (Modified BSD 3 clause license)
          Specifically, modifications must be indicated.
          We have a modified version of MITK here:
          https://github.com/NifTK/MITK.git
          
Library : Slicer Execution Model 
Website : https://github.com/Slicer/SlicerExecutionModel
Purpose : Part of the build system for building command line apps
License : Documentation/Licenses/SlicerExecutionModel (Slicer license)
          
Library : Qt
Website : http://qt.nokia.com/products
Purpose : Gui framework.
License : Documentation/Licenses/Qt.txt (LGPL v2.1 license)

Library : CTK
Website : http://www.commontk.org
Purpose : Common medical imaging specific classes.
License : Documentation/Licenses/CTK_LICENSE.txt (Apache v2.0 license)

Library : NiftyReg
Website : https://cmiclab.cs.ucl.ac.uk/mmodat/niftyreg
Purpose : Image Registration Library
License : Documentation/Licenses/NiftyReg.txt (BSD 3 clause license)

Library : NiftySeg
Website : http://sourceforge.net/projects/niftyseg/ 
Purpose : Image Segmentation Library
License : Documentation/Licenses/NiftySeg.txt (BSD 3 clause license)

Library : NiftySim
Website : http://sourceforge.net/projects/niftysim/
Purpose : Non-Linear Finite Element Solver Library
License : Documentation/Licenses/NiftySim.txt (BSD 3 clause license)

Library : NiftyLink
Website : https://cmiclab.cs.ucl.ac.uk/CMIC/NiftyLink 
Purpose : Messaging Library to talk to client applications
License : Documentation/Licenses/NiftyLink.txt (BSD 3 clause license)

Library : NiftyCal
Website : https://cmiclab.cs.ucl.ac.uk/CMIC/NiftyCal
Purpose : Video camera calibration
License : https://cmiclab.cs.ucl.ac.uk/CMIC/NiftyCal/blob/master/LICENSE.txt (BSD 3 clause license)

