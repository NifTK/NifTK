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

#include "niftkSurfReconExports.h"
#include <mitkDataStorage.h>
#include <mitkImage.h>
#include <itkObject.h>
#include <itkObjectFactoryBase.h>
#include <mitkGenericProperty.h>
//#include <opencv2/core/core.hpp>
#include <itkMatrix.h>
#include <niftkUndistortion.h>

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
class NIFTKSURFRECON_EXPORT SurfaceReconstruction : public itk::Object
{
public:
  enum Method
  {
    SEQUENTIAL_CPU          = 0,
    PYRAMID_PARALLEL_CPU    = 1,
    PYRAMID_PARALLEL_CUDA
  };

  /**
   * Returns some details of a available reconstruction methods.
   * You should loop from zero in parameter index until it returns false.
   * Stuff the returned friendlyname into a GUI combobox, and when the user
   * chooses a method, read it back from the combobox and pass it into ParseMethodName()
   * to retrieve the ID that can be passed into Run().
   */
  static bool GetMethodDetails(int index, Method* id, std::string* friendlyname);

  /**
   * Will throw an exception of the name is not recognised.
   */
  static Method ParseMethodName(const std::string& friendlyname);


  enum OutputType
  {
    MITK_POINT_CLOUD  = 1,
    PCL_POINT_CLOUD   = 2,    // BEWARE: may not be compiled in!
    DISPARITY_IMAGE   = 3
  };

public:

  mitkClassMacro(SurfaceReconstruction, itk::Object);
  itkNewMacro(SurfaceReconstruction);

  /**
   * @warning niftk::Undistortion::s_StereoRigTransformationPropertyName is taken from either image1 or image2
   *          but it is interpreted as coming from image2! This is a bug, I think.
   *
   * @throws std::logic_error if method is not implemented (beware: just because a method ID exists in 
   *                          the Method enum doesn't mean it's actually implemented! you need to loop
   *                          through GetMethodDetails().)
   * @throws std::runtime_error if the dimensions of image1 and image2 are not the same.
   * @throws std::runtime_error if outputtype is POINT_CLOUD and either image lacks property niftk::Undistortion::s_CameraCalibrationPropertyName.
   * @throws std::runtime_error if property niftk::Undistortion::s_CameraCalibrationPropertyName is not of type mitk::CameraIntrinsicsProperty
   * @throws std::runtime_error if property niftk::Undistortion::s_StereoRigTransformationPropertyName is not of type niftk::MatrixProperty
   * @throws std::runtime_error if MITK image accessors fail.
   */
  mitk::BaseData::Pointer Run(
           const mitk::Image::Pointer image1,
           const mitk::Image::Pointer image2,
           Method method,
           OutputType outputtype,
           const mitk::DataNode::Pointer camnode,
           float maxTriangulationError,
           float minDepth,
           float maxDepth,
           bool bakeCameraTransform);

  /** Exists mainly for the benefit of uk.ac.ucl.cmic.igisurfacerecon plugin. */
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
    bool bakeCameraTransform;

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
