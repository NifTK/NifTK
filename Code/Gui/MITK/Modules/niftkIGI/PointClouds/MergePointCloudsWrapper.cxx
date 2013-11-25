/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "MergePointCloudsWrapper.h"
#include <mitkPointSetReader.h>


namespace niftk
{


//-----------------------------------------------------------------------------
MergePointCloudsWrapper::MergePointCloudsWrapper()
  : m_MergedPointSet(mitk::PointSet::New())
{
}


//-----------------------------------------------------------------------------
MergePointCloudsWrapper::~MergePointCloudsWrapper()
{
}


//-----------------------------------------------------------------------------
void MergePointCloudsWrapper::AddPointSet(const std::string& filename)
{
  if (filename.empty())
    throw std::runtime_error("Point cloud file name cannot be empty");

  // read .mps file with mitk's build-in mechanism (very slow).
  mitk::PointSetReader::Pointer   psreader = mitk::PointSetReader::New();
  psreader->SetFileName(filename);
  psreader->Update();
  if (!psreader->GetSuccess())
    throw std::runtime_error("Could not read point set file " + filename);

  mitk::PointSet::Pointer pointset = psreader->GetOutput();
  assert(pointset.IsNotNull());

  AddPointSet(pointset);
}


//-----------------------------------------------------------------------------
void MergePointCloudsWrapper::AddPointSet(const mitk::PointSet::Pointer& pointset)
{
  AddPointSet(mitk::PointSet::ConstPointer(pointset.GetPointer()));
}


//-----------------------------------------------------------------------------
void MergePointCloudsWrapper::AddPointSet(const mitk::PointSet::ConstPointer& pointset)
{
  // we are allocating ids sequentially, so this should be fine.
  unsigned int  id = m_MergedPointSet->GetSize();
  for (mitk::PointSet::PointsConstIterator i = pointset->Begin(); i != pointset->End(); ++i, ++id)
  {
    assert(m_MergedPointSet->IndexExists(id) == false);

    const mitk::PointSet::PointType& p = i->Value();
    m_MergedPointSet->InsertPoint(id, mitk::PointSet::PointType(&p[0]));
  }

}


//-----------------------------------------------------------------------------
mitk::PointSet::Pointer MergePointCloudsWrapper::GetOutput() const
{
  return m_MergedPointSet->Clone();
}


} // namespace
