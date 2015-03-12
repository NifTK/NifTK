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
  Common/mitkImage2DToTexturePlaneMapper3D.cxx
  Common/mitkNifTKIGIObjectFactory.cxx
  DataSources/mitkIGITestDataUtils.cxx
  DataSources/mitkIGIDataType.cxx
  DataSources/mitkIGIDataSource.cxx
  DataSources/mitkIGIOpenCVDataType.cxx
  SurfaceReconstruction/SurfaceReconstruction.cxx
  SurfaceReconstruction/SequentialCpuQds.cxx
  SurfaceReconstruction/QDSCommon.cxx
  TrackedImage/mitkTrackedImage.cxx
  TrackedPointer/mitkTrackedPointer.cxx
  PointBasedRegistration/mitkPointBasedRegistration.cxx
  SurfaceBasedRegistration/mitkSurfaceBasedRegistration.cxx
  Utils/mitkMakeGeometry.cxx
  # this one does not depend on pcl!
  PointClouds/mitkMergePointClouds.cxx
  CentreLines/mitkBifurcationToPointSet.cxx
  Rendering/vtkCalibratedModelRenderingPipeline.cxx
)

if(BUILD_PCL)
  list(APPEND CPP_FILES
    PCLTest/mitkPCLTest.cxx
    PointClouds/FitPlaneToPointCloudWrapper.cxx
    PointClouds/mitkPCLData.cxx
  )
endif()
