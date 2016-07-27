/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkMergePointClouds_h
#define niftkMergePointClouds_h

#include "niftkCoreExports.h"
#include <string>
#include <ostream>
#include <mitkCommon.h>
#include <mitkPointSet.h>
#include <itkObject.h>
#include <itkObjectFactory.h>
#include <itkObjectFactoryBase.h>

namespace niftk
{

/// Will not preserve point IDs in the merged output!
/// Order of points is not necessarily preserved.
class NIFTKCORE_EXPORT MergePointClouds : public itk::Object
{
public:
  mitkClassMacroItkParent(MergePointClouds, itk::Object)
  itkNewMacro(MergePointClouds)

protected:
  MergePointClouds();
  virtual ~MergePointClouds();

  /// Not implemented
  MergePointClouds(const MergePointClouds&);
  /// Not implemented
  MergePointClouds& operator=(const MergePointClouds&);

public:
  /**
   * @throws std::runtime_error if filename is empty
   * @throws std::runtime_error if the file cannot be read/parsed
   * @throws mitk::Exception if the file does not exist
   */
  void AddPointSet(const std::string& filename);

  /// @throws std::runtime_error if pointset is null.
  void AddPointSet(const mitk::PointSet::Pointer& pointset);

  /// @throws std::runtime_error if pointset is null.
  void AddPointSet(const mitk::PointSet::ConstPointer& pointset);

  /// @throws nothing should not throw anything
  mitk::PointSet::Pointer GetOutput() const;

private:
  mitk::PointSet::Pointer       m_MergedPointSet;
};

}

#endif
