/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __mitkLabelMapReader_h
#define __mitkLabelMapReader_h


#include <mitkAbstractFileReader.h>

#include "mitkLabelMapWriter.h"

#include <QColor.h>
#include <QString.h>

namespace mitk 

{

/**
* /brief Reader for label map files. 
* /ingroup IO
*/
class NIFTKCOREGUI_EXPORT LabelMapReader : public AbstractFileReader
{

public: 
  LabelMapReader();
  LabelMapReader(const LabelMapReader & other);
  virtual LabelMapReader * Clone() const;
  virtual ~LabelMapReader(){};
  
  using mitk::AbstractFileReader::Read;
  virtual std::vector<itk::SmartPointer<BaseData> > Read();  
  
  inline std::vector< LabelMapItem > GetLabels(){ return m_Labels; }

private: 
  void ReadLabelMap();  
  
  std::vector< LabelMapItem > m_Labels;
  
  us::ServiceRegistration<mitk::IFileReader> m_ServiceReg;

};

} // namespace mitk

#endif // __mitkLabelMapReader_h