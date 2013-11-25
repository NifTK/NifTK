/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef mitkMergePointClouds_h
#define mitkMergePointClouds_h


#include "niftkIGIExports.h"
#include <string>
#include <ostream>
#include <mitkCommon.h>
#include <mitkPointSet.h>
#include <itkObject.h>
#include <itkObjectFactory.h>
#include <itkObjectFactoryBase.h>


namespace mitk
{


class NIFTKIGI_EXPORT MergePointClouds : public itk::Object
{
public:
  mitkClassMacro(MergePointClouds, itk::Object);
  itkNewMacro(MergePointClouds);


protected:
  /** Not implemented */
  MergePointClouds();
  /** Not implemented */
  virtual ~MergePointClouds();

  /** Not implemented */
  MergePointClouds(const MergePointClouds&);
  /** Not implemented */
  MergePointClouds& operator=(const MergePointClouds&);


public:
  void AddPointSet(const std::string& filename);
  void AddPointSet(const mitk::PointSet::Pointer& pointset);
  void AddPointSet(const mitk::PointSet::ConstPointer& pointset);

  mitk::PointSet::Pointer GetOutput() const;

private:
  mitk::PointSet::Pointer       m_MergedPointSet;
};


} // namespace


#endif // mitkMergePointClouds_h
