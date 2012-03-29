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
  QString boostVersion(NIFTK_BOOST_VERSION);
  QString gdcmVersion(NIFTK_GDCM_VERSION);
  QString itkVersion(NIFTK_ITK_VERSION);
  QString vtkVersion(NIFTK_VTK_VERSION);
  QString qtVersion(NIFTK_QT_VERSION);
  QString ctkVersion(NIFTK_CTK_VERSION);
  QString mitkVersion(NIFTK_MITK_VERSION);
  QString svnVersion(NIFTK_SVN_VERSION);

  // Main titles with application name, release version and copyright statement.
  QString titles = QObject::tr(
      "<h1>About %1</h1>"
      "<h3>%2</h3>"
      "<p>%3.</p>"
      ).arg(applicationName).arg(versionNumber).arg(copyrightText);

  // Short introduction.
  QString introduction = QObject::tr(
      "<p>"
      "%1 is the user interface for the <a href=\"%2\">%3 (%4)</a> translational imaging platform called %5."
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
      "<a href=\"http://www.mitk.org\">MITK</a> and <a href=\"http://www.commontk.org\">CTK</a>."
      "In addition, various clip art comes from <a href=\"http://www.openclipart.org\">openclipart.org</a>"
      "</p>"
      ).arg(originShortText);

  // Over time, insert more software packages, as platform expands,
  // (and dependencies get exponentially more frustrating :-).
  QString versions = QObject::tr(
      "<h3>Software Versions</h3>"
      "<p>"
      "%1 has been developed using the following libraries"
      "<ul>"
      "<li><a href=\"http://www.boost.org\">Boost</a>:%2</li>"
      "<li><a href=\"http://qt.nokia.com/products\">Qt</a>:%6</li>"
      "<li><a href=\"http://www.creatis.insa-lyon.fr/software/public/Gdcm/\">GDCM</a>:%3</li>"
      "<li><a href=\"http://www.itk.org\">ITK</a>:%4</li>"
      "<li><a href=\"http://www.vtk.org\">VTK</a>:%5</li>"
      "<li><a href=\"http://www.commontk.org\">CTK</a>:%7</li>"
      "<li><a href=\"http://www.mitk.org\">MITK</a>:%8</li>"
      "</ul>"
      "which themselves depend on further libraries. This version of %9 was built with our subversion revision <a href=\"https://cmicdev.cs.ucl.ac.uk/trac/browser/trunk/NifTK\">%10</a>"
      "</p>"
      )
      .arg(applicationName)
      .arg(boostVersion)
      .arg(gdcmVersion)
      .arg(itkVersion)
      .arg(vtkVersion)
      .arg(qtVersion)
      .arg(ctkVersion.left(10))
      .arg(mitkVersion.left(10))
      .arg(applicationName)
      .arg(svnVersion);

  // Over time, insert more platforms that we have tested on,
  // (but these should be backed up with a Dashboard or else it ain't worth diddly-squat).
  QString testingDetails = QObject::tr(
      "<p>"
      "%1 has been compiled and tested on:"
      "<ul>"
      "<li>Mac OSX 10.7</li>"
      "<li>Mac OSX 10.6</li>"
      "<li>Scientific Linux 6.1</li>"
      "<li>Centos 5.0</li>"
      "<li>Ubuntu 11.04</li>"
      "<li>Windows 7</li>"
      "<li>Windows XP</li>"
      "</ul>"
      "and our software quality control statistics can be seen on this <a href=\"%2\">%3</a>."
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
      .append(versions)
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
