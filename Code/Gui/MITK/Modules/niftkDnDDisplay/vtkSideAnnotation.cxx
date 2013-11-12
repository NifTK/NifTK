/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "vtkSideAnnotation_p.h"

#include <vtkObjectFactory.h>
#include <vtkTextProperty.h>
#include <vtkTextMapper.h>

#include <cstring>


//----------------------------------------------------------------------------
vtkStandardNewMacro(vtkSideAnnotation);


//----------------------------------------------------------------------------
vtkSideAnnotation::vtkSideAnnotation()
{
  for (int i = 0; i < 4; i++)
  {
    this->TextMapper[i]->GetTextProperty()->GetColor(m_Colours[i]);
  }
}


//----------------------------------------------------------------------------
vtkSideAnnotation::~vtkSideAnnotation()
{
}


//----------------------------------------------------------------------------
void vtkSideAnnotation::SetTextActorsPosition(int vsize[2])
{
  this->TextActor[0]->SetPosition(vsize[0] / 2.0, vsize[1] - 5.0);
  this->TextActor[1]->SetPosition(vsize[0] - 5.0, vsize[1] / 2.0);
  this->TextActor[2]->SetPosition(vsize[0] / 2.0, 5.0);
  this->TextActor[3]->SetPosition(5.0, vsize[1] / 2.0);
}


//----------------------------------------------------------------------------
void vtkSideAnnotation::SetTextActorsJustification()
{
  vtkTextProperty* textProperty = this->TextMapper[0]->GetTextProperty();
  textProperty->SetJustificationToCentered();
  textProperty->SetVerticalJustificationToTop();

  textProperty = this->TextMapper[1]->GetTextProperty();
  textProperty->SetJustificationToRight();
  textProperty->SetVerticalJustificationToCentered();

  textProperty = this->TextMapper[2]->GetTextProperty();
  textProperty->SetJustificationToCentered();
  textProperty->SetVerticalJustificationToBottom();

  textProperty = this->TextMapper[3]->GetTextProperty();
  textProperty->SetJustificationToLeft();
  textProperty->SetVerticalJustificationToCentered();
}


//----------------------------------------------------------------------------
int vtkSideAnnotation::RenderOpaqueGeometry(vtkViewport* viewport)
{
  int result = Superclass::RenderOpaqueGeometry(viewport);

  // Note that the superclass has restored the common text properties
  // to each text mappers. Therefore, we set the text colours now.

  for (int i = 0; i < 4; ++i)
  {
    this->TextMapper[i]->GetTextProperty()->SetColor(m_Colours[i]);
  }

  return result;
}


//----------------------------------------------------------------------------
void vtkSideAnnotation::SetColour(int i, double* colour)
{
  if (i < 0 || i > 3)
  {
    return;
  }

  std::memcpy(m_Colours[i], colour, sizeof(m_Colours[i]));

  // Note that we do not change the text property here, because the Modified()
  // call triggers the call of RenderOpaqueGeometry that restores the original,
  // common colour to each text mappers.
  // We change the colour of the texts after the rendering, instead.
  // See vtkSideAnnotation::RenderOpaqueGeometry(vtkViewport*).

  this->Modified();
}


//----------------------------------------------------------------------------
void vtkSideAnnotation::GetColour(int i, double* colour)
{
  if (i < 0 || i > 3)
  {
    return;
  }

  std::memcpy(colour, m_Colours[i], sizeof(m_Colours[i]));
}
