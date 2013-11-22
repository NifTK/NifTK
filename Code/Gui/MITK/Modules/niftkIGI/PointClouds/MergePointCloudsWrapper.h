/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef MergePointCloudsWrapper_h
#define MergePointCloudsWrapper_h


#include "niftkIGIExports.h"
#include <string>
#include <ostream>
#include <mitkCommon.h>
#include <mitkPointSet.h>
#include <itkObject.h>
#include <itkObjectFactory.h>
#include <itkObjectFactoryBase.h>


namespace niftk
{


// i dont think this should be called ...Wrapper. suggestions for a better name?
class NIFTKIGI_EXPORT MergePointCloudsWrapper : public itk::Object
{
public:
  mitkClassMacro(MergePointCloudsWrapper, itk::Object);
  itkNewMacro(MergePointCloudsWrapper);


protected:
  /** Not implemented */
  MergePointCloudsWrapper();
  /** Not implemented */
  virtual ~MergePointCloudsWrapper();

  /** Not implemented */
  MergePointCloudsWrapper(const MergePointCloudsWrapper&);
  /** Not implemented */
  MergePointCloudsWrapper& operator=(const MergePointCloudsWrapper&);


public:
  void AddPointSet(const std::string& filename);
  void AddPointSet(const mitk::PointSet::Pointer& pointset);
  void AddPointSet(const mitk::PointSet::ConstPointer& pointset);

  // FIXME: should this return a clone?
  mitk::PointSet::ConstPointer GetOutput() const;

private:
  mitk::PointSet::Pointer       m_MergedPointSet;
};


} // namespace


#endif // MergePointCloudsWrapper
