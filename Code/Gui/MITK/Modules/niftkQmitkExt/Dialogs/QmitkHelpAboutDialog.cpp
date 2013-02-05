/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-12-16 09:12:58 +0000 (Fri, 16 Dec 2011) $
 Revision          : $Revision: 8039 $
 Last modified by  : $Author: mjc $

 Original author   : a.duttaroy@cs.ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#include "QmitkHelpAboutDialog.h"

#include <QMessageBox>
#include <QString>
#include <iostream>
#include "NifTKConfigure.h"

QmitkHelpAboutDialog::QmitkHelpAboutDialog(QWidget *parent, QString applicationName)
                : m_ApplicationName(applicationName)
{
  if(!m_ApplicationName.isEmpty())
  {
	  this->setupUi(this);
	  this->GenerateHelpAboutText(m_ApplicationName);
	  this->m_HelpAboutLabel->setWordWrap(true);
	  this->m_HelpAboutLabel->setOpenExternalLinks(true);
	  this->m_HelpAboutLabel->setTextFormat(Qt::RichText);

  }
}

QmitkHelpAboutDialog::~QmitkHelpAboutDialog()
{

}

void QmitkHelpAboutDialog::GenerateHelpAboutText(QString applicationName)
{
  // This stuff gets generated during CMake into NifTKConfigure.h
  QString platformName(NIFTK_PLATFORM);
  QString versionNumber(NIFTK_VERSION_STRING);
  QString copyrightText(NIFTK_COPYRIGHT);
  QString originURL(NIFTK_ORIGIN_URL);
  QString originShortText(NIFTK_ORIGIN_SHORT_TEXT);
  QString originLongText(NIFTK_ORIGIN_LONG_TEXT);
  QString wikiURL(NIFTK_WIKI_URL);
  QString wikiText(NIFTK_WIKI_TEXT);
  QString dashboardURL(NIFTK_DASHBOARD_URL);
  QString dashboardText(NIFTK_DASHBOARD_TEXT);
  QString userContact(NIFTK_USER_CONTACT);
  QString developerContact(NIFTK_DEVELOPER_CONTACT);
  QString qtVersion(NIFTK_QT_VERSION);
  QString boostVersion(NIFTK_BOOST_VERSION);
  QString gdcmVersion(NIFTK_GDCM_VERSION);
  QString dcmtkVersion(NIFTK_DCMTK_VERSION);
  QString itkVersion(NIFTK_ITK_VERSION);
  QString vtkVersion(NIFTK_VTK_VERSION);
  QString ctkVersion(NIFTK_CTK_VERSION);
  QString mitkVersion(NIFTK_MITK_VERSION);
  QString niftkVersion(NIFTK_VERSION);
  QString niftkDateTime(NIFTK_DATE_TIME);
  QString boostLocation(NIFTK_BOOST_LOCATION);
  QString gdcmLocation(NIFTK_GDCM_LOCATION);
  QString dcmtkLocation(NIFTK_DCMTK_LOCATION);
  QString itkLocation(NIFTK_ITK_LOCATION);
  QString vtkLocation(NIFTK_VTK_LOCATION);
  QString ctkLocation(NIFTK_CTK_LOCATION);
  QString mitkLocation(NIFTK_MITK_LOCATION);
#ifdef USE_NIFTYREC
  QString niftyRecVersion(NIFTK_NIFTYREC_VERSION);
  QString niftyRecLocation(NIFTK_NIFTYREC_LOCATION);
#endif
#ifdef USE_NIFTYSIM
  QString niftySimVersion(NIFTK_NIFTYSIM_VERSION);
  QString niftySimLocation(NIFTK_NIFTYSIM_LOCATION);
#endif

  // Main titles with application name, release version and copyright statement.
  QString titles = QObject::tr(
      "<h1>About %1</h1>"
      "<h3>%2</h3>"
      "<p>%3.</p>"
      ).arg(applicationName).arg(versionNumber).arg(copyrightText);

  // Short introduction.
  QString introduction = QObject::tr(
      "<p>"
      "%1 is the user interface for the <a href=\"%2\">%3 (%4)</a> translational imaging platform called <a href=\"http://cmic.cs.ucl.ac.uk/home/software/\">%5</a>."
      "</p>"
      "<p>"
      "%1 was developed with funding from the NIHR and the Comprehensive Biomedical Research Centre at UCL and UCLH grant 168 and TSB grant M1638A. "
      "The principal investigator is <a href=\"http://cmic.cs.ucl.ac.uk/staff/sebastien_ourselin/\">Sebastien Ourselin</a> "
      "and team leader is <a href=\"http://cmic.cs.ucl.ac.uk/staff/matt_clarkson/\">Matt Clarkson</a>."
      "</p>"
      ).arg(applicationName).arg(originURL).arg(originLongText).arg(originShortText).arg(platformName);

  // Over time, insert more collaborators, as we conquer the world!!
  // (mwah ha ha ha .. evil laughter).
  QString collaborators = QObject::tr(
      "<p>"
      "%1 is grateful for the continued support of our clinical and research collaborators including:"
      "<ul>"
      "<li>the <a href=\"http://dementia.ion.ucl.ac.uk/\">UCL Dementia Research Centre</a>.</li>"
      "<li>the <a href=\"http://www.ucl.ac.uk/ion/departments/neuroinflammation/\">UCL Department of Neuroinflammation</a>.</li>"
      "<li>the <a href=\"http://www.ucl.ac.uk/cabi/\">UCL Centre for Advanced Biomedical Imaging</a>.</li>"
      "</ul>"
      "In addition, the software development team would like to acknowledge the kind support of the open-source software community "
      "during development of NifTK and are especially grateful to the developers of "
      "<a href=\"http://www.mitk.org\">MITK</a> and <a href=\"http://www.commontk.org\">CTK</a>. "
      "In addition, various clip art comes from <a href=\"http://www.openclipart.org\">openclipart.org</a>."
      "</p>"
      ).arg(originShortText);

  // Over time, insert more software packages, as platform expands,
  // (and dependencies get exponentially more frustrating :-).
  QString versionsStart = QObject::tr(
      "<h3>Software Versions</h3>"
      "<p>"
      "%1 has been developed using the following core libraries."
      "</p>"
      "<p><table>"
      "<tr><td><a href=\"http://www.boost.org\">Boost</a></td><td>%2</td><td><a href=\"http://www.boost.org/LICENSE_1_0.txt\">Boost software license version 1.0</a></td><td><a href=\"%3\">from here</a></td></tr>"
      "<tr><td><a href=\"http://qt.nokia.com/products\">Qt</a></td><td>%4</td><td><a href=\"http://www.gnu.org/licenses/old-licenses/lgpl-2.1.html\">LGPL version 2.1</a></td><td><a href=\"http://qt.nokia.com/products\">from here</a></td></tr>"
      "<tr><td><a href=\"http://www.creatis.insa-lyon.fr/software/public/Gdcm/\">GDCM</a></td><td>%5</td><td><a href=\"http://www.creatis.insa-lyon.fr/software/public/Gdcm/License.html\">GDCM Berkely-like license</a></td><td><a href=\"%6\">from here</a></td></tr>"
      "<tr><td><a href=\"http://dicom.offis.de/\">DCMTK</a></td><td>%7</td><td><a href=\"ftp://dicom.offis.de/pub/dicom/offis/software/dcmtk/dcmtk360/COPYRIGHT\">DCMTK license</a></td><td><a href=\"%8\">from here</a></td></tr>"
      "<tr><td><a href=\"http://www.itk.org\">ITK</a></td><td>%9</td><td><a href=\"http://www.itk.org/ITK/project/licenseversion2.html\">Simplified BSD license for version 3.20</a></td><td><a href=\"%10\">from here</a></td></tr>"
      "<tr><td><a href=\"http://www.vtk.org\">VTK</a></td><td>%11</td><td><a href=\"http://www.vtk.org/VTK/project/license.html\">BSD license</a></td><td><a href=\"%12\">from here</a></td></tr>"
      "<tr><td><a href=\"http://www.commontk.org\">CTK</a></td><td>%13</td><td><a href=\"http://www.apache.org/licenses/LICENSE-2.0.html\">Apache 2.0 license</a></td><td><a href=\"%14\">from here</a></td></tr>"
      "<tr><td><a href=\"http://www.mitk.org\">MITK</a>(Modified)</td><td>%15</td><td><a href=\"http://www.mitk.org/wiki/License\">BSD-style license</a></td><td><a href=\"%16\">from here</a></td></tr>"
      )
      .arg(applicationName)
      .arg(boostVersion)
      .arg(boostLocation)
      .arg(qtVersion)
      .arg(gdcmVersion)
      .arg(gdcmLocation)
      .arg(dcmtkVersion)
      .arg(dcmtkLocation)
      .arg(itkVersion)
      .arg(itkLocation)
      .arg(vtkVersion)
      .arg(vtkLocation)
      .arg(ctkVersion.left(10))
      .arg(ctkLocation)
      .arg(mitkVersion.left(10))
      .arg(mitkLocation)
      ;

  #ifdef USE_NIFTYREG
    QString niftyRegVersion(NIFTK_NIFTYREG_VERSION);
    QString niftyRegLocation(NIFTK_NIFTYREG_LOCATION);
    QString niftyRegText = QObject::tr(
        "<tr><td><a href=\"http://www0.cs.ucl.ac.uk/staff/M.Modat/Marcs_Page/Software.html\">NiftyReg</a></td><td>%1</td><td><a href=\"http://niftyreg.svn.sourceforge.net/viewvc/niftyreg/trunk/nifty_reg/LICENSE.txt?revision=1&view=markup\">BSD license</a></td><td><a href=\"%2\">from here</a></td></tr>"
        ).arg(niftyRegVersion).arg(niftyRegLocation);
  #endif

  #ifdef USE_NIFTYSEG
    QString niftySegVersion(NIFTK_NIFTYSEG_VERSION);
    QString niftySegLocation(NIFTK_NIFTYSEG_LOCATION);
    QString niftySegText = QObject::tr(
        "<tr><td><a href=\"http://niftyseg.sourceforge.net\">NiftySeg</a></td><td>%1</td><td><a href=\"http://niftyseg.sourceforge.net/Documentation/styled-3/index.html\">BSD license</a></td><td><a href=\"%2\">from here</a></td></tr>"
        ).arg(niftySegVersion).arg(niftySegLocation);
  #endif

  #ifdef BUILD_IGI
    QString niftyLinkVersion(NIFTK_NIFTYLINK_VERSION);
    QString niftyLinkLocation(NIFTK_NIFTYLINK_LOCATION);
    QString niftyLinkText = QObject::tr(
      "<tr><td><a href=\"https://cmicdev.cs.ucl.ac.uk/NiftyLink/html/index.html\">NiftyLink</a></td><td>%1</td><td><a href=\"https://cmicdev.cs.ucl.ac.uk/NiftyLink/html/NiftyLinkLicense.html\">Not finalised yet</a></td><td><a href=\"%2\">from here</a></td></tr>"
      ).arg(niftyLinkVersion.left(10)).arg(niftyLinkLocation);
  #endif

  QString versionsEnd = QObject::tr(
      "</table></p>"
      );

  QString licenses = QObject::tr(
      "<p>"
      "The licenses can be found online and are additionally included in the installation folder. This version of %1 was built with our git hash <a href=\"https://cmicdev.cs.ucl.ac.uk/trac/browser/niftk\">%2</a>, from %3."
      "</p>"
      ).arg(applicationName).arg(niftkVersion).arg(niftkDateTime);

  // Over time, insert more platforms that we have tested on,
  // (but these should be backed up with a Dashboard or else it ain't worth diddly-squat).
  QString testingDetails = QObject::tr(
      "<p>"
      "%1 has been compiled and tested on the following platforms:"
      "<ul>"
      "<li>Mac OSX 10.7 (Lion)</li>"
      "<li>Mac OSX 10.6 (Snow Leopard)</li>"
      "<li>Ubuntu 11.04 (Natty)</li>"
      "<li>Linux Mint 12</li>"
      "<li>Scientific Linux 6.1</li>"
      "<li>Centos 5.0</li>"
      "<li>Debian 6.0.5</li>"
      "<li>Windows 7</li>"
      "<li>Windows XP</li>"
      "</ul>"
      "We assume a 64 bit operating system. Our software quality control statistics can be seen on this <a href=\"%2\">%3</a>."
      "</p>"
      ).arg(applicationName).arg(dashboardURL).arg(dashboardText);

  QString furtherInformation = QObject::tr(
      "<p>"
      "Further information can be obtained by:"
      "<ul>"
      "<li>Emailing the %1 <a href=\"%2\">users mailing list</a>.</li>"
      "<li>Emailing the %1 <a href=\"%3\">developers mailing list</a>.</li>"
      "<li>Visiting the %1 <a href=\"%4\">%5</a>.</li>"
      "</ul>"
      "</p>"
      ).arg(platformName).arg(userContact).arg(developerContact).arg(wikiURL).arg(wikiText);

  // Stick it all together.
  QString totalText =
      titles
      .append(introduction)
      .append(collaborators)
      .append(furtherInformation)
      .append(versionsStart)
#ifdef USE_NIFTYREG
      .append(niftyRegText)
#endif
#ifdef USE_NIFTYSEG
      .append(niftySegText)
#endif
#ifdef BUILD_IGI
      .append(niftyLinkText)
#endif
      .append(versionsEnd)
      .append(licenses)
      .append(testingDetails)
      ;

  this->setWindowTitle(tr("About %1").arg(applicationName));
  QIcon helpAboutIcon(QLatin1String(":/Icons/icon.png"));

  if (!helpAboutIcon.isNull())
  {
    this->setWindowIcon(helpAboutIcon);
  }

  this->m_HelpAboutLabel->setText(totalText);

}
