/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "QmitkIGIOverlayEditor.h"

//-----------------------------------------------------------------------------
QmitkIGIOverlayEditor::QmitkIGIOverlayEditor(QWidget * /*parent*/)
{
  this->setupUi(this);
}


//-----------------------------------------------------------------------------
QmitkIGIOverlayEditor::~QmitkIGIOverlayEditor()
{
}


//-----------------------------------------------------------------------------
void QmitkIGIOverlayEditor::SetDataStorage(const mitk::DataStorage* storage)
{

}


//-----------------------------------------------------------------------------
QmitkRenderWindow* QmitkIGIOverlayEditor::GetActiveQmitkRenderWindow() const
{

}


//-----------------------------------------------------------------------------
QHash<QString, QmitkRenderWindow *> QmitkIGIOverlayEditor::GetQmitkRenderWindows() const
{

}


//-----------------------------------------------------------------------------
QmitkRenderWindow* QmitkIGIOverlayEditor::GetQmitkRenderWindow(const QString &id) const
{

}


//-----------------------------------------------------------------------------
mitk::Point3D QmitkIGIOverlayEditor::GetSelectedPosition(const QString &id) const
{

}


//-----------------------------------------------------------------------------
void QmitkIGIOverlayEditor::SetSelectedPosition(const mitk::Point3D &pos, const QString &id)
{

}


//-----------------------------------------------------------------------------
void QmitkIGIOverlayEditor::SetDepartmentLogoPath(const std::string path)
{

}


//-----------------------------------------------------------------------------
void QmitkIGIOverlayEditor::EnableDepartmentLogo()
{

}


//-----------------------------------------------------------------------------
void QmitkIGIOverlayEditor::DisableDepartmentLogo()
{

}


//-----------------------------------------------------------------------------
void QmitkIGIOverlayEditor::SetGradientBackgroundColors(const mitk::Color& colour1, const mitk::Color& colour2)
{

}


//-----------------------------------------------------------------------------
void QmitkIGIOverlayEditor::EnableGradientBackground()
{

}


//-----------------------------------------------------------------------------
void QmitkIGIOverlayEditor::DisableGradientBackground()
{

}
