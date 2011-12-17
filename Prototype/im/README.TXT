/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: $
 Revision          : $Revision: $
 Last modified by  : $Author: $

 Original author   : malone@drc.ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

Files:
itk-sphere-connect.cpp
itk-sphere-label.cpp
itk-sphere.cpp
vtk-sphere-label.cpp
detailed-label-spec
labelspec2
simple-spec
simple-spec-2
testprofile.txt

itk-sphere-connect.cpp
Add connections to labelled vtk surface, based on profiles or default 2 region connections.

itk-sphere-label.cpp
vtk-sphere-label.cpp
Apply labels to a vtk surface, assuming a sphere either apply regions specified by sphere locations and distances or a default splitting into octants. vtk-sphere-label is preferred as it preserves edges (i.e. can be run on connected spheres) while the itk version can't.

itk-sphere.cpp
Generate a subdivided icosahedron spherical mesh.

detailed-label-spec
36 region (1:36) specification attempting to achieve the level of complexity from FreeSurfer cortical surfaces.

simple-spec-2regions.txt
Two hemisphere description (0:1)

labelspec2
simple-spec
simple-spec-2
Several low region count profiles.

testprofile.txt
Test 7*7 profile
