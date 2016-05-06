/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __niftkLabelMapWriter_h
#define __niftkLabelMapWriter_h



#include <mitkAbstractFileWriter.h>
#include "niftkLabeledLookupTableProperty.h"



namespace niftk 
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
class LabelMapWriter : public mitk::AbstractFileWriter
{

public: 

  LabelMapWriter();
  virtual ~LabelMapWriter();

  virtual void Write() override;  

private: 

  LabelMapWriter(const LabelMapWriter & other);
  virtual LabelMapWriter * Clone() const override;

  /** \brief Write labels, lookuptable to stream. */
  void WriteLabelMap(
    LabeledLookupTableProperty::LabelListType labels,
    vtkLookupTable* lookupTable) const; 
};

} // namespace mitk

#endif
