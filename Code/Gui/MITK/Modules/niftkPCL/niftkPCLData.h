/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkPCLData_h
#define niftkPCLData_h

#include "niftkPCLExports.h"
#include <mitkBaseData.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>


namespace niftk
{

class NIFTKPCL_EXPORT PCLData : public mitk::BaseData
{
public:
  mitkClassMacro(PCLData, mitk::BaseData);

  itkFactorylessNewMacro(Self);


  /** @name MITK stuff not applicable. */
  //@{
  /** Does nothing. */
  virtual void SetRequestedRegionToLargestPossibleRegion();
  /** @returns false always */
  virtual bool RequestedRegionIsOutsideOfTheBufferedRegion();
  /** @returns true always */
  virtual bool VerifyRequestedRegion();
  /** Does nothing. */
  virtual void SetRequestedRegion(const itk::DataObject* data);
  //@}


  void SetCloud(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud);
  pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr GetCloud() const;


protected:
  PCLData();
  virtual ~PCLData();


  /** @name Copy and assignment not allowed. */
  //@{
private:
  PCLData(const PCLData& copyme);
  PCLData& operator=(const PCLData& assignme);
  //@}


private:
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr    m_Cloud;
};


} // namespace

#endif
