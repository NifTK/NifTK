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
#include <vtkIntArray.h>
#include <vtkStringArray.h>

namespace mitk
{
	
//-----------------------------------------------------------------------------
vtkLookupTable* ChangeColor(vtkLookupTable* lut, int value, const QColor& newColor)
{
  vtkLookupTable* newLUT = vtkLookupTable::New();
  newLUT->DeepCopy(lut);

  vtkIdType index = newLUT->GetIndex(value);
  newLUT->SetTableValue(index, newColor.redF(), newColor.greenF(), newColor.blueF(), newColor.alphaF());

  return newLUT;
}


//-----------------------------------------------------------------------------
vtkLookupTable* SwapColors(vtkLookupTable* lut, int value1, int value2)
{
  vtkLookupTable* newLUT;
  // if either index is not in bounds, resize table
  double* range = lut->GetRange();
  if (value1 > range[1] || value2 > range[1])
  {
    double maxValue = std::max(value1, value2);
    newLUT = ResizeLookupTable(lut, maxValue + 1);
  }
  else
  {
    newLUT = vtkLookupTable::New();
    newLUT->DeepCopy(lut);
  }
  
  vtkIdType index1 = newLUT->GetIndex(value1);
  vtkIdType index2 = newLUT->GetIndex(value2);

  //one of the values does not exist in the annotations -- so reinitialize
  if (index1 == -1) 
  {
    vtkSmartPointer<vtkIntArray> annotationValueArray = dynamic_cast<vtkIntArray*>(newLUT->GetAnnotatedValues());
    vtkSmartPointer<vtkStringArray> annotationNameArray = newLUT->GetAnnotations();

    annotationValueArray->InsertValue(value1, value1);
    annotationNameArray->InsertValue(value1, "");
    newLUT->SetAnnotations(annotationValueArray, annotationNameArray);
  }

  if (index2 == -1)
  {
    vtkSmartPointer<vtkIntArray> annotationValueArray = dynamic_cast<vtkIntArray*>(newLUT->GetAnnotatedValues());
    vtkSmartPointer<vtkStringArray> annotationNameArray = newLUT->GetAnnotations();

    annotationValueArray->InsertValue(value2, value2);
    annotationNameArray->InsertValue(value2, "");
    newLUT->SetAnnotations(annotationValueArray, annotationNameArray);
  }

  double rgba1[4];
  newLUT->GetIndexedColor(index1, rgba1);

  double rgba2[4];
  newLUT->GetIndexedColor(index2, rgba2);

  newLUT->SetTableValue(value1, rgba2);
  newLUT->SetTableValue(value2, rgba1);

  return newLUT;
}


//-----------------------------------------------------------------------------
vtkLookupTable* ResizeLookupTable(vtkLookupTable* lut, double newMaximum)
{
  vtkLookupTable* newLUT = vtkLookupTable::New();

  newLUT->DeepCopy(lut);

  double newRange[2];
  newRange[0] = 0;
  newRange[1] = newMaximum;

  newLUT->SetRange(newRange); // this automatically invalidates the old colors so we need to explicitly set them
  int numberOfColors = newMaximum + 1;
  newLUT->SetNumberOfColors(numberOfColors);

  newLUT->Build();

  for (int i = 0; i < lut->GetNumberOfColors(); i++)
  {
    double rgba[4];
    lut->GetTableValue(i, rgba);
    newLUT->SetTableValue(i, rgba);

    // make sure we set the annotated values
    if(newLUT->GetAnnotatedValueIndex(i) == -1)
    {
      newLUT->SetAnnotation(i, "");
    }
  }

  double rgba[4];
  lut->GetTableValue(lut->GetNumberOfColors(), rgba);
  for (unsigned int j = lut->GetNumberOfColors(); j < newLUT->GetNumberOfColors(); j++)
  {
    newLUT->SetTableValue(j, rgba);
    newLUT->SetAnnotation(j, "");
  }

  return newLUT;
}

vtkLookupTable* CreateEmptyLookupTable(const QColor& lowColor, const QColor& highColor)
{
  vtkLookupTable* lookupTable = vtkLookupTable::New();
  lookupTable->SetNumberOfColors(1);
  lookupTable->SetTableRange(0, 0);
  lookupTable->SetValueRange(lowColor.value(), highColor.value());
  lookupTable->SetHueRange(lowColor.hue(), highColor.hue());
  lookupTable->SetSaturationRange(lowColor.saturation(), highColor.saturation());
  lookupTable->SetAlphaRange(lowColor.alpha(), highColor.alpha());
  lookupTable->SetNanColor(0, 0, 0, highColor.alpha());
  lookupTable->SetIndexedLookup(true);  

  lookupTable->Build();

  return lookupTable;
}

} //namespace mitk
