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

#ifndef __PNMReader_h
#define __PNMReader_h

#include <mitkCommon.h>
#include <mitkFileReader.h>
#include <vtkSmartPointer.h>

#include <mitkAbstractFileReader.h>

namespace mitk
{

  /** \brief
  */

  class PNMReader : public AbstractFileReader
  {
  public:

    PNMReader();
    virtual ~PNMReader(){}
    PNMReader(const PNMReader& other);
    virtual PNMReader * Clone() const;

    using mitk::AbstractFileReader::Read;
    virtual std::vector<itk::SmartPointer<BaseData> > Read();

  private:

    us::ServiceRegistration<mitk::IFileReader> m_ServiceReg;
  };

} //namespace MITK

#endif // __PNMReader_h
