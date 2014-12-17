/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include <niftkCSVRow.h>

namespace niftk
{
  CSVRow::CSVRow()
  {
    m_data = new std::vector<std::string>();
  }

  CSVRow::~CSVRow()
  {
    delete m_data;
  }

  void CSVRow::ReadNextRow( std::istream& inStream )
  {
    std::string strRowRead;
    std::getline( inStream, strRowRead );

    std::stringstream ssRow( strRowRead );
    std::string strElement;

    m_data->clear();
    while( std::getline( ssRow, strElement, ',' ) )
    {
      m_data->push_back( strElement );
    }
  }

} // end namespace
