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
    assert(pointset->IndexExists(id) == false);

    const mitk::PointSet::PointType& p = i->Value();
    m_MergedPointSet->InsertPoint(id, mitk::PointSet::PointType(&p[0]));
  }

}


} // namespace


/*
int main(int argc, char* argv[])
{
  if (argc < 1)
  {
    std::cerr << "specify one or more mitk mps pointset file(s)!" << std::endl;
    return 1;
  }

  std::string   firstfilename(argv[1]);

  unsigned int    id = 0;
  mitk::PointSet::Pointer   output = mitk::PointSet::New();
  for (int i = 1; i < argc; ++i)
  {
    std::string   mpsfilename(argv[i]);
    std::cerr << "Reading pointset from file: " << mpsfilename;

    mitk::PointSetReader::Pointer   psreader = mitk::PointSetReader::New();
    psreader->SetFileName(mpsfilename);
    psreader->Update();
    mitk::PointSet::Pointer pointset = psreader->GetOutput();
    std::cerr << "...done" << std::endl;

    for (mitk::PointSet::PointsConstIterator i = pointset->Begin(); i != pointset->End(); ++i, ++id)
    {
      const mitk::PointSet::PointType& p = i->Value();
      output->InsertPoint(id, mitk::PointSet::PointType(&p[0]));
    }
  }

  // dont bother with mitk's pointset writer. it's ridiculously slow.
  std::ofstream   mpsfile(firstfilename + ".merged.mps");
  mpsfile << "<?xml version=\"1.0\" encoding=\"ISO-8859-1\"?>\n<point_set_file>\n<file_version>0.1</file_version>\n<point_set><time_series><time_series_id>0</time_series_id>" << std::endl;
  for (mitk::PointSet::PointsConstIterator i = output->Begin(); i != output->End(); ++i)
  {
    const mitk::PointSet::PointType& p = i->Value();
    mpsfile << "<point><id>" << i->Index() << "</id><specification>0</specification>"
            << "<x>" << p[0] << "</x><y>" << p[1] << "</y><z>" << p[2] << "</z></point>" << std::endl;
  }
  mpsfile << "</time_series></point_set></point_set_file>" << std::endl;
  mpsfile.close();

}
  */
