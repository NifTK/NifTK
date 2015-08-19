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
vtkLookupTable* ChangeColor(vtkLookupTable* lut, int value, QColor newColor)
{

  vtkLookupTable* newLUT = vtkLookupTable::New();
  newLUT->DeepCopy(lut);
  newLUT->SetNanColor(lut->GetNanColor());

  vtkIdType index = newLUT->GetIndex(value);
  newLUT->SetTableValue(index, newColor.redF(), newColor.greenF(), newColor.blueF() );

  return newLUT;
}


//-----------------------------------------------------------------------------
vtkLookupTable* SwapColors(vtkLookupTable* lut, int value1, int value2)
{
  vtkLookupTable* newLUT;
  // if either index is not in bounds, resize table
  double* range = lut->GetRange();
  if( value1<range[0] || value2<range[0] || value1>range[1] || value2>range[1] )
  {
    double newRange[2];
    double minValue = std::min(value1,value2);
    newRange[0] = std::min(range[0], minValue-1);

    double maxValue = std::max(value1,value2);
    newRange[1] = std::max(range[1],maxValue+1);
    newLUT = ResizeLookupTable(lut,newRange);
  }
  else
  {
    newLUT = vtkLookupTable::New();
    newLUT->DeepCopy(lut);
    newLUT->SetNanColor(lut->GetNanColor());
  }
  vtkIdType index1 = newLUT->GetIndex(value1);
  double rgba1[4];
  newLUT->GetIndexedColor(index1,rgba1);

  vtkIdType index2 = newLUT->GetIndex(value2);
  double rgba2[4];
  newLUT->GetIndexedColor(index2,rgba2);

  newLUT->SetTableValue(index1,rgba2);
  newLUT->SetTableValue(index2,rgba1);

  return newLUT;
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

vtkLookupTable* CreateEmptyLookupTable()
{
  
  vtkLookupTable* lookupTable = vtkLookupTable::New();
  lookupTable->SetValueRange(0,0);
  lookupTable->SetHueRange(0,0);
  lookupTable->SetSaturationRange(0,0);
  lookupTable->SetAlphaRange(0,0);
  lookupTable->SetNanColor(0,0,0,0);
  lookupTable->Build();

  return lookupTable;
}
