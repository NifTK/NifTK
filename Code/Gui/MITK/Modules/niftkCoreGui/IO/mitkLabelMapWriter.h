/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __mitkLabelMapWriter_h
#define __mitkLabelMapWriter_h



#include <mitkAbstractFileWriter.h>

#include "niftkCoreGuiExports.h"
#include "mitkLabelMapReader.h"
#include "QmitkLookupTableContainer.h"

#include <QColor>

namespace mitk 

{


/**
* /brief Reader for label map files. 
* /ingroup IO
*/
class NIFTKCOREGUI_EXPORT LabelMapWriter : public AbstractFileWriter
{

public: 
  typedef QmitkLookupTableContainer::LabelsListType LabelsListType;

  LabelMapWriter();
  LabelMapWriter(const LabelMapWriter & other);
  virtual LabelMapWriter * Clone() const;
  virtual ~LabelMapWriter(){};
  
  using mitk::AbstractFileWriter::Write;
  virtual void Write();  
  
  inline void SetLabels(std::vector< LabelMapItem >  labels){ m_Labels = labels; }

  void SetLabelsAndLookupTable(LabelsListType labels, vtkLookupTable* vtkLUT);

  void WriteLabelMap();  

private: 
  
  std::vector< LabelMapItem > m_Labels;
  

};

} // namespace mitk

#endif