/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef XnatBrowserView_h
#define XnatBrowserView_h

#include "ui_XnatBrowserView.h"

#include <berryQtViewPart.h>
#include <QmitkAbstractView.h>
#include <mitkDataNode.h>

class QWidget;

class XnatBrowserViewPrivate;

/**
 * \class XnatBrowserView
 * \brief Provides a simple GUI for browsing XNAT databases
 * \ingroup uk_ac_ucl_cmic_xnat_internal
 */
class XnatBrowserView : public QmitkAbstractView
{

  // this is needed for all Qt objects that should have a MOC object (everything that derives from QObject)
  Q_OBJECT

public:

  explicit XnatBrowserView();
  virtual ~XnatBrowserView();

  /// \brief Each View for a plugin has its own globally unique ID, this one is
  /// "uk.ac.ucl.cmic.xnat.browser" and the .cxx file and plugin.xml should match.
  static const QString VIEW_ID;

protected:

  /// \brief Called by framework, this method creates all the controls for this view
  virtual void CreateQtPartControl(QWidget *parent) override;

  /// \brief Called by framework, sets the focus on a specific widget.
  virtual void SetFocus() override;

private:

  /// \brief All the controls for the main view part.
  Ui::XnatBrowserView* m_Controls;

  // Store a reference to the parent widget of this view.
  QWidget *m_Parent;

  /// \brief d pointer of the pimpl pattern
  QScopedPointer<XnatBrowserViewPrivate> d_ptr;

  Q_DECLARE_PRIVATE(XnatBrowserView);
  Q_DISABLE_COPY(XnatBrowserView);
};
#endif // _XNATBROWSERVIEW_H_INCLUDED
