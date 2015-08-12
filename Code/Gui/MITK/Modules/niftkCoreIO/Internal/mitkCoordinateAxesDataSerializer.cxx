/*===================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center,
Division of Medical and Biological Informatics.
All rights reserved.

This software is distributed WITHOUT ANY WARRANTY; without
even the implied warranty of MERCHANTABILITY or FITNESS FOR
A PARTICULAR PURPOSE.

See LICENSE.txt or http://www.mitk.org for details.

===================================================================*/

#include "mitkCoordinateAxesDataSerializer.h"
#include "mitkCoordinateAxesData.h"
#include "niftkCoordinateAxesDataWriterService.h"

#include <itksys/SystemTools.hxx>


MITK_REGISTER_SERIALIZER(CoordinateAxesDataSerializer)


mitk::CoordinateAxesDataSerializer::CoordinateAxesDataSerializer()
{
}


mitk::CoordinateAxesDataSerializer::~CoordinateAxesDataSerializer()
{
}


std::string mitk::CoordinateAxesDataSerializer::Serialize()
{
  const CoordinateAxesData* image = dynamic_cast<const CoordinateAxesData*>( m_Data.GetPointer() );
  if (image == NULL)
  {
    MITK_ERROR << " Object at " << (const void*) this->m_Data
              << " is not an mitk::CoordinateAxesData. Cannot serialize as CoordinateAxesData.";
    return "";
  }

  std::string filename( this->GetUniqueFilenameInWorkingDirectory() );
  filename += "_";
  filename += m_FilenameHint;
  filename += ".4x4";

  std::string fullname(m_WorkingDirectory);
  fullname += "/";
  fullname += itksys::SystemTools::ConvertToOutputPath(filename.c_str());

  try
  {
    niftk::CoordinateAxesDataWriterService writer;
    writer.SetOutputLocation(fullname);
    writer.SetInput(const_cast<CoordinateAxesData*>(image));
    writer.Write();
  }
  catch (std::exception& e)
  {
    MITK_ERROR << " Error serializing object at " << (const void*) this->m_Data
              << " to "
              << fullname
              << ": "
              << e.what();
    return "";
  }
  return filename;
}

