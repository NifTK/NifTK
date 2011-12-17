/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-11-24 16:50:16 +0000 (Thu, 24 Nov 2011) $
 Revision          : $Revision: 7860 $
 Last modified by  : $Author: mjc $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
 
#ifndef ThumbnailView_h
#define ThumbnailView_h

#include "berryQtViewPart.h"
#include "ui_ThumbnailViewControls.h"

/**
 * \class ThumbnailView
 * \brief Provides a thumbnail view of the currently focused QmitkRenderWindow.
 * \ingroup uk.ac.ucl.cmic.thumbnail_internal
 *
 * Note: This class should just be a wrapper that instantiates the
 * widget QmitkThumnailRenderWindow, and does almost nothing else.
 * Do not try and add loads more functionality here.
*/
class ThumbnailView : public berry::QtViewPart
{  
  // this is needed for all Qt objects that should have a Qt meta-object
  // (everything that derives from QObject and wants to have signal/slots)
  Q_OBJECT
  
public:

  ThumbnailView();
  virtual ~ThumbnailView();

  /// \brief Static view ID = uk.ac.ucl.cmic.thumbnailview
  static const std::string VIEW_ID;

  /// \brief Called from framework to instantiate the Qt GUI components.
  virtual void CreateQtPartControl(QWidget *parent);

  /// \brief Required implementation from berry::QtViewPart
  virtual void SetFocus() {}

private:

  // All the controls for the main view part.
  Ui::ThumbnailViewControls *m_Controls;

};

#endif // ThumbnailView_h

