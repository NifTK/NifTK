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
#include "niftkCoreGuiExports.h"


#include <QColor.h>
#include <QString.h>
#include <qfile.h>

class QmitkLookupTableContainer;

namespace mitk 

{
struct LabelMapItem
{
  double  value;
  QColor  color;
  QString name;
};

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
  
  // get the mitk base data from a file
  virtual std::vector<itk::SmartPointer<BaseData> > Read();  
  
  inline std::vector< LabelMapItem > GetLabels(){ return m_Labels; }
  
  // get a QmitkLookupTableContainer from a file
  virtual QmitkLookupTableContainer* GetLookupTableContainer();
 
  /** Set the order to list the label map in, since label map files do not store this information */
  void SetOrder(int order){m_Order = order;}

  void SetQFile(QFile &file){m_InputQFile = &file;};

private: 

  bool ReadLabelMap(std::istream &);  
  
  std::vector< LabelMapItem > m_Labels;
  
  us::ServiceRegistration<mitk::IFileReader> m_ServiceReg;

  int m_Order;

  QString m_DisplayName;

  QFile* m_InputQFile;
};

} // namespace mitk

#endif // __mitkLabelMapReader_h