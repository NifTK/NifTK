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
#include "mitkLabeledLookupTableProperty.h"



namespace mitk 
{


/**
  * \brief Writer to save labeled lookup tables in the format of Slicer v 4.4.0.  
  *
  * The labels are assumed to correspond to the range of the 
  * vtkLookupTable. Out of range label values will be 
  * assigned a color according to GetTableValue().
  *
  * \ingroup IO
  */
class NIFTKCOREGUI_EXPORT LabelMapWriter : public AbstractFileWriter
{

public: 

  typedef mitk::LabeledLookupTableProperty::LabelListType LabelListType;

  LabelMapWriter();
  LabelMapWriter(const LabelMapWriter & other);
  virtual LabelMapWriter * Clone() const;
  virtual ~LabelMapWriter(){};
  
  using mitk::AbstractFileWriter::Write;

  /** \brief Write labels, lookuptable to file.*/
  virtual void Write();  

  /** \brief Set the labels to write to file. */
  inline void SetLabels(const LabelListType& labels){m_Labels = labels;}

  /** \brief Set the vtkLookupTable to write to file. */
  inline void SetVtkLookupTable(vtkLookupTable* vtkLUT){m_LookupTable = vtkLUT;}

private: 

  /** \brief Write labels, lookuptable to stream. */
  void WriteLabelMap(); 

  /** To store the labels to write to file. */
  LabelListType m_Labels;

  /** To store the vtkLookupTable to write to file. */
  vtkLookupTable* m_LookupTable;
};

} // namespace mitk

#endif
