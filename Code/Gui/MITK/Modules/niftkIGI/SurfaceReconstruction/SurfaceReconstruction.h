/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef SurfaceReconstruction_h
#define SurfaceReconstruction_h

#include "niftkIGIExports.h"
#include <mitkDataStorage.h>
#include <mitkImage.h>
#include <itkObject.h>
#include <itkObjectFactoryBase.h>
#include <mitkGenericProperty.h>
//#include <opencv2/core/core.hpp>
#include <itkMatrix.h>
#include <CameraCalibration/Undistortion.h>


// forward-decl
namespace niftk 
{
class SequentialCpuQds;
}


namespace niftk 
{


// used for stereo-rig transformation, i.e. between left and right camera
// FIXME: sticking in an opencv matrix would be prefered
typedef mitk::GenericProperty<itk::Matrix<float, 4, 4> >    MatrixProperty;


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
    SEQUENTIAL_CPU          = 0,
    PYRAMID_PARALLEL_CPU    = 1,
    PYRAMID_PARALLEL_CUDA
  };


  enum OutputType
  {
    POINT_CLOUD,
    DISPARITY_IMAGE
  };

public:

  mitkClassMacro(SurfaceReconstruction, itk::Object);
  itkNewMacro(SurfaceReconstruction);

  /**
   * \brief Write My Documentation
   */
  mitk::BaseData::Pointer Run(
           const mitk::Image::Pointer image1,
           const mitk::Image::Pointer image2,
           Method method,
           OutputType outputtype,
           const mitk::DataNode::Pointer camnode,
           float maxTriangulationError,
           float minDepth,
           float maxDepth);


  struct ParamPacket
  {
    mitk::DataStorage::Pointer dataStorage;
    mitk::DataNode::Pointer outputNode;
    mitk::Image::Pointer image1;
    mitk::Image::Pointer image2;
    Method method;
    OutputType outputtype;
    mitk::DataNode::Pointer camnode;
    float maxTriangulationError;
    float minDepth;
    float maxDepth;

    ParamPacket()
    {
    }
  };

  mitk::BaseData::Pointer Run(ParamPacket params);

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
