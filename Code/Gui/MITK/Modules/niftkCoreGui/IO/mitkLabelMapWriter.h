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
#include "QmitkLookupTableContainer.h"



namespace mitk 

{


/**
  * /brief Writer to save labeled lookup tables in the format of Slicer v 4.4.0.  
  *
  * The labels are assumed to correspond to the range of the 
  * vtkLookupTable. Out of range label values will be 
  * assigned a color according to GetTableValue().
  *
  * /ingroup IO
  */
class NIFTKCOREGUI_EXPORT LabelMapWriter : public AbstractFileWriter
{

public: 
  typedef QmitkLookupTableContainer::LabelListType LabelListType;

  LabelMapWriter();
  LabelMapWriter(const LabelMapWriter & other);
  virtual LabelMapWriter * Clone() const;
  virtual ~LabelMapWriter(){};
  
  using mitk::AbstractFileWriter::Write;
  virtual void Write();  
  
  inline void SetLabels(LabelListType labels){m_Labels = labels;}
  inline void SetVtkLookupTable(vtkLookupTable* vtkLUT){ m_LookupTable = vtkLUT;}

  void WriteLabelMap();  

private: 

  LabelListType m_Labels;
  vtkLookupTable* m_LookupTable;
  
};

} // namespace mitk

#endif