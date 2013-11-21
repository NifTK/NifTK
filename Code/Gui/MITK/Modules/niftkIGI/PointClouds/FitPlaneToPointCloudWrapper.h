/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef FitPlaneToPointCloudWrapper_h
#define FitPlaneToPointCloudWrapper_h


#include "niftkIGIExports.h"
#include <string>
#include <ostream>
#include <mitkCommon.h>
#include <mitkPointSet.h>
#include <itkObject.h>
#include <itkObjectFactory.h>
#include <itkObjectFactoryBase.h>


// forward declaration to avoid pulling in truckloads of pcl headers.
namespace pcl
{
struct ModelCoefficients;
}


namespace niftk
{


// i dont think this should be called ...Wrapper. suggestions for a better name?
class NIFTKIGI_EXPORT FitPlaneToPointCloudWrapper : public itk::Object
{
public:
  mitkClassMacro(FitPlaneToPointCloudWrapper, itk::Object);
  itkNewMacro(FitPlaneToPointCloudWrapper);


protected:
  /** Not implemented */
  FitPlaneToPointCloudWrapper();
  /** Not implemented */
  virtual ~FitPlaneToPointCloudWrapper();

  /** Not implemented */
  FitPlaneToPointCloudWrapper(const FitPlaneToPointCloudWrapper&);
  /** Not implemented */
  FitPlaneToPointCloudWrapper& operator=(const FitPlaneToPointCloudWrapper&);


public:
  /**
   * @throws std::runtime_error if filename is emty
   * @throws std::runtime_error if file cannot be read (or parsed) for whatever reason
   * @throws std::runtime_error if the point cloud in the file has less than 4 points
   * @throws std::runtime_error if a plane could not be estimated
   */
  void FitPlane(const std::string& filename);

  /**
   * @throws std::runtime_error if pointset has less than 4 points
   * @throws std::runtime_error if a plane could not be estimated
   */
  void FitPlane(const mitk::PointSet::Pointer& pointset);
  void PrintOutput(std::ostream& log) const;

  /**
   * @throws std::logic_error if FitPlane() has not been called
   */
  void GetParameters(float& a, float& b, float& c, float& d) const;


private:
  pcl::ModelCoefficients*     m_PlaneCoefficients;

  float   m_MinPlaneDistance;
  float   m_MaxPlaneDistance;
  float   m_AvgPlaneDistance;
  float   m_RmsPlaneDistance;
};


} // namespace

#endif // FitPlaneToPointCloudWrapper
