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

#include "QmitkMIDASBaseFunctionality.h"
#include "berryIPartListener.h"
#include "berryIPreferences.h"
#include "berryIPreferencesService.h"
#include "berryIBerryPreferences.h"
#include "berryISelection.h"
#include "berryISelectionProvider.h"
#include "berryISelectionListener.h"
#include "ui_ThumbnailViewControls.h"
/**
 * \class ThumbnailView
 * \brief Provides a thumbnail view of the currently focused QmitkRenderWindow.
 * \ingroup uk.ac.ucl.cmic.thumbnail_internal
 *
 * Note: This class should basically just be a wrapper that instantiates the
 * widget QmitkThumnailRenderWindow, and does almost nothing else.
 * Do not try and add loads more functionality here, just the necessary plumbing.
 *
 * \sa QmitkThumbnailRenderWindow
*/
class ThumbnailView : public QmitkMIDASBaseFunctionality
{  
  // this is needed for all Qt objects that should have a Qt meta-object
  // (everything that derives from QObject and wants to have signal/slots)
  Q_OBJECT
  
public:

  berryObjectMacro(ThumbnailView);
  ThumbnailView();
  virtual ~ThumbnailView();

  /// \brief Static view ID = uk.ac.ucl.cmic.thumbnailview
  static const std::string VIEW_ID;

  /// \brief Returns the view ID.
  virtual std::string GetViewID() const;

  /// \brief Called from framework to instantiate the Qt GUI components.
  virtual void CreateQtPartControl(QWidget *parent);

  /// \brief Called when preferences are updated.
  virtual void OnPreferencesChanged(const berry::IBerryPreferences*);

  /// \brief This is not an exclusive functionality, as it just listens to input and updates itself, and can happily live alongside other functionalities.
  virtual bool IsExclusiveFunctionality() const { return false; }

private:

  /// \brief Retrieve's the pref values from preference service, and store locally.
  void RetrievePreferenceValues();

  // All the controls for the main view part.
  Ui::ThumbnailViewControls *m_Controls;
};

#endif // ThumbnailView_h

