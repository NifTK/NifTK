/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkSurfaceReconstruction_h
#define niftkSurfaceReconstruction_h

#include "niftkIGIExports.h"
#include <mitkDataStorage.h>
#include <mitkImage.h>
#include <itkObject.h>
#include <itkObjectFactoryBase.h>


namespace niftk 
{


// forward-decl
class SequentialCpuQds;


/**
 * \class SurfaceReconstruction
 * \brief Takes image data, and calculates a surface reconstruction, and write a point cloud to mitk::DataStorage.
 */
class NIFTKIGI_EXPORT SurfaceReconstruction : public itk::Object
{
public:
  // The order of these should match the entries in SurfaceReconViewWidget::MethodComboBox,
  // or better the other way around: combobox needs to match this order.
  enum Method
  {
    SEQUENTIAL_CPU,
    PYRAMID_PARALLEL_CPU,
    PYRAMID_PARALLEL_CUDA
  };


  enum OutputType
  {
    POINT_CLOUD,
    DISPARITY_IMAGE
  };


  // FIXME: i dont think this is the best place to keep these. i'm up for suggestions!
  static const char*    s_ImageIsUndistortedPropertyName;   // mitk::BoolProperty
  static const char*    s_ImageIsRectifiedPropertyName;     // mitk::BoolProperty
  static const char*    s_CameraCalibrationPropertyName;    // mitk::CameraIntrinsicsProperty


public:

  mitkClassMacro(SurfaceReconstruction, itk::Object);
  itkNewMacro(SurfaceReconstruction);

  /**
   * \brief Write My Documentation
   */
  void Run(const mitk::DataStorage::Pointer dataStorage,
           mitk::DataNode::Pointer outputNode,
           const mitk::Image::Pointer image1,
           const mitk::Image::Pointer image2,
           Method method = SEQUENTIAL_CPU,
           OutputType outputtype = POINT_CLOUD);

protected:

  SurfaceReconstruction(); // Purposefully hidden.
  virtual ~SurfaceReconstruction(); // Purposefully hidden.

  SurfaceReconstruction(const SurfaceReconstruction&); // Purposefully not implemented.
  SurfaceReconstruction& operator=(const SurfaceReconstruction&); // Purposefully not implemented.

private:
  SequentialCpuQds*    m_SequentialCpuQds;

}; // end class

} // end namespace

#endif // niftkSurfaceReconstruction_h
