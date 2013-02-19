/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "XnatBrowserView.h"

#include <berryIPreferencesService.h>
#include <berryIBerryPreferences.h>

#include "XnatBrowserWidget.h"
#include "XnatPluginPreferencePage.h"
#include "XnatPluginSettings.h"

const std::string XnatBrowserView::VIEW_ID = "uk.ac.ucl.cmic.xnat.browser";

class XnatBrowserViewPrivate
{
public:
  XnatPluginSettings* settings;
  XnatBrowserWidget* xnatBrowserWidget;
};

XnatBrowserView::XnatBrowserView()
: m_Controls(0)
, m_Parent(0)
, d_ptr(new XnatBrowserViewPrivate())
{
}

XnatBrowserView::~XnatBrowserView()
{
  Q_D(XnatBrowserView);

  delete d->settings;

//  if (m_Controls)
//  {
//    delete m_Controls;
//  }
}

void XnatBrowserView::CreateQtPartControl(QWidget *parent)
{
  Q_D(XnatBrowserView);

  // setup the basic GUI of this view
  m_Parent = parent;

  d->settings = new XnatPluginSettings(GetPreferences());

  d->xnatBrowserWidget = new XnatBrowserWidget(parent);
  QVBoxLayout* layout = new QVBoxLayout();
  layout->addWidget(d->xnatBrowserWidget);
  parent->setLayout(layout);
  d->xnatBrowserWidget->setSettings(d->settings);
  d->xnatBrowserWidget->setDataStorage(GetDataStorage());

//  if (!m_Controls)
//  {
//    // Create UI
//    m_Controls = new Ui::XnatBrowserView();
//    m_Controls->setupUi(parent);
//
//    m_Controls->xnatBrowserWidget->setSettings(d->settings);
//    m_Controls->xnatBrowserWidget->setDataStorage(GetDataStorage());
//  }
}

void XnatBrowserView::SetFocus()
{
  Q_D(XnatBrowserView);
//  m_Controls->xnatBrowserWidget->setFocus();
  d->xnatBrowserWidget->setFocus();
}
