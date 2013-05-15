/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/
 
#ifndef ThumbnailView_h
#define ThumbnailView_h

#include <QmitkAbstractView.h>
#include <berryIPartListener.h>
#include <berryIPreferences.h>
#include <berryIPreferencesService.h>
#include <berryIBerryPreferences.h>
#include <berryISelection.h>
#include <berryISelectionProvider.h>
#include <berryISelectionListener.h>
#include "ui_ThumbnailViewControls.h"
/**
 * \class ThumbnailView
 * \brief Provides a thumbnail view of the currently focused QmitkRenderWindow.
 * \ingroup uk.ac.ucl.cmic.thumbnail_internal
 *
 * Note: This class should basically just be a wrapper that instantiates the
 * widget QmitkThumnailRenderWindow, and does almost nothing else.
 * Do not try and add loads more functionality here, just do the necessary plumbing.
 *
 * \sa QmitkThumbnailRenderWindow
*/
class ThumbnailView : public QmitkAbstractView
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

  /// \brief Called when preferences are updated.
  virtual void OnPreferencesChanged(const berry::IBerryPreferences*);

  /// \brief This is not an exclusive functionality, as it just listens to input and updates itself, and can happily live alongside other functionalities.
  virtual bool IsExclusiveFunctionality() const { return false; }

protected:

  /// \brief Called by framework, this method creates all the controls for this view
  virtual void CreateQtPartControl(QWidget *parent);

  /// \brief Called by framework, sets the focus on a specific widget.
  virtual void SetFocus();

private:

  /// \brief Retrieve's the pref values from preference service, and store locally.
  void RetrievePreferenceValues();

  // All the controls for the main view part.
  Ui::ThumbnailViewControls *m_Controls;
};

#endif // ThumbnailView_h

