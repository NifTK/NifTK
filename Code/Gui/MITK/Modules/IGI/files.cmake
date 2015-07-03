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

set(CPP_FILES
  mitkNifTKIGIObjectFactory.cxx
  TrackedImage/mitkTrackedImage.cxx
  TrackedPointer/mitkTrackedPointer.cxx
  Utils/mitkMakeGeometry.cxx
  PointClouds/mitkMergePointClouds.cxx
  CentreLines/mitkBifurcationToPointSet.cxx
  Rendering/mitkImage2DToTexturePlaneMapper3D.cxx
  Rendering/vtkCalibratedModelRenderingPipeline.cxx
  MicroServices/niftkPointRegServiceI.cxx
  MicroServices/niftkPointRegServiceRAII.cxx
)

