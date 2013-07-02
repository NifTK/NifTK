/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include <niftkMIDASGuiExports.h>

#include "vtkCornerAnnotation.h"

/**
 * \class vtkSideAnnotation
 * \brief Subclass of vtkCornerAnnotation to display annotations on the four sides
 * of a render window, instead of its corners. Additionally, the class supports
 * setting different colours for each annotations, individually.
 *
 * The sides are numbered from 0 to 3 in the following order: top, right, bottom, left.
 */
class NIFTKMIDASGUI_EXPORT vtkSideAnnotation : public vtkCornerAnnotation
{
public:
  vtkTypeMacro(vtkSideAnnotation, vtkCornerAnnotation);

  static vtkSideAnnotation* New();

  /// \brief Overrides vtkCornerAnnotation::RenderOpaqueGeometry(vtkViewPort*) to
  /// restore the colours after the annotations are re-rendered.
  int RenderOpaqueGeometry(vtkViewport *viewport);

  /// \brief Sets the colour of the text on the specified side.
  void SetColour(int i, double* colour);

  /// \brief Gets the colour of the text on the specified side.
  void GetColour(int i, double* colour);

protected:

  /// \brief Constructs the vtkSideAnnotation object.
  vtkSideAnnotation();
  /// \brief Destructs the vtkSideAnnotation object.
  virtual ~vtkSideAnnotation();

  /// \brief Overrides vtkCornerAnnotation::SetTextActorsPosition(int vsize[2])
  /// to position the annotations on the sides instead of the corners.
  virtual void SetTextActorsPosition(int vsize[2]);

  /// \brief Overrides vtkCornerAnnotation::SetTextActorsJustification()
  /// to align the text to the inner side of the render window.
  virtual void SetTextActorsJustification();

private:

  /// \brief The colour of each text annotation.
  double m_Colours[4][3];

  vtkSideAnnotation(const vtkSideAnnotation&);  // Purposefully not implemented.
  void operator=(const vtkSideAnnotation&);  // Purposefully not implemented.
};
