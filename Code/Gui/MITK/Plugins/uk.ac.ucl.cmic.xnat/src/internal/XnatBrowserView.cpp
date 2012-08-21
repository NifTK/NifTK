/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-07-19 17:52:47 +0100 (Tue, 19 Jul 2011) $
 Revision          : $Revision: 6804 $
 Last modified by  : $Author: mjc $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

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
