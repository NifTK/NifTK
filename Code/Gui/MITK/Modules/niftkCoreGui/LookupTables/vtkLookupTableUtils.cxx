/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/



#include "vtkLookupTableUtils.h"
#include <vtkLookupTable.h>
#include <vtkSmartPointer.h>
#include <qcolor.h>


//-----------------------------------------------------------------------------
void ChangeColor(vtkLookupTable* lut, int value, QColor newColor)
{
  vtkIdType index = lut->GetIndex(value);
  lut->SetTableValue(index, newColor.redF(), newColor.greenF(), newColor.blueF() );
}


//-----------------------------------------------------------------------------
void SwapColors(vtkLookupTable* lut, int value1, int value2)
{

  vtkIdType index1 = lut->GetIndex(value1);
  double rgba1[4];
  lut->GetIndexedColor(index1,rgba1);

  vtkIdType index2 = lut->GetIndex(value2);
  double rgba2[4];
  lut->GetIndexedColor(index2,rgba2);

  lut->SetTableValue(index1,rgba2);
  lut->SetTableValue(index2,rgba1);
}


//-----------------------------------------------------------------------------
vtkLookupTable* ResizeLookupTable(vtkLookupTable* lut, double* newRange)
{
  vtkLookupTable* newLUT = vtkLookupTable::New();

  newLUT->DeepCopy(lut);
  newLUT->SetNanColor(lut->GetNanColor());

  
  newLUT->SetRange(newRange); // this automatically invalidates the old colors so we need to explicitly set them
  int numberOfColors = newRange[1]-newRange[0];
  newLUT->SetNumberOfColors(numberOfColors);

  newLUT->Build();

  for(int i=0;i< lut->GetNumberOfColors();i++)
  {
    double rgba[4];
    lut->GetTableValue(i, rgba);
    newLUT->SetTableValue(i,rgba);
  }

  for(unsigned int j=lut->GetNumberOfColors();j<newLUT->GetNumberOfColors();j++)
  {
    newLUT->SetTableValue(j, lut->GetNanColor());
  }
  return newLUT;
}