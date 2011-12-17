/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-08-18 12:23:46 +0100 (Thu, 18 Aug 2011) $
 Revision          : $Revision: 7128 $
 Last modified by  : $Author: ad $

 Original author   : a.duttaroy@cs.ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
#ifndef HELPABOUTDIALOG_CPP
#define HELPABOUTDIALOG_CPP

#include <QMessageBox>
#include <QString>
#include "HelpAboutDialog.h"
#include "NifTKConfigure.h"


//Constructor
HelpAboutDialog::HelpAboutDialog(QWidget *parent, QString applicationName)
                : m_ApplicationName(applicationName)
{
  if(!m_ApplicationName.isEmpty())
  {
	  this->setupUi(this);
    SetHelpAboutText(m_ApplicationName);
  }
}

//Destructor
HelpAboutDialog::~HelpAboutDialog()
{

}

void HelpAboutDialog::SetHelpAboutText(QString applicationName)
{
  QString versionNumber(NIFTK_VERSION_STRING);

  QString translatedTextAboutQtCaption;
  translatedTextAboutQtCaption = QMessageBox::tr(
      "<h1>About %1</h1>"
      "<h3>%2</h3>"
      "<p>Copyright &copy; 2008-2011 <a href=\"http://www.ucl.ac.uk/\">University College London</a></p>"
      ).arg(applicationName).arg(versionNumber);

  QString translatedTextAboutQtText;
  translatedTextAboutQtText = QMessageBox::tr(
      "<p>"
      "%1 is the user interface for the <a href=\"http://cmic.cs.ucl.ac.uk/\">Centre For Medical Image Computing (CMIC's)</a> translational imaging platform"
      "</p>"
      "<p>"
      "%1 was developed with funding from the NIHR and the Comprehensive Biomedical Research Centre at UCL and UCLH grant 168 and TSB grant M1638A."
      "The principal investigator is <a href=\"http://cmic.cs.ucl.ac.uk/staff/sebastien_ourselin/\">Sebastien Ourselin</a> and team leader is <a href=\"http://cmic.cs.ucl.ac.uk/staff/matt_clarkson/\">Matt Clarkson</a>"
      "</p>"
      "<p>"
      "The aim is to deliver software tools into clinical settings, and CMIC is grateful to all our clinical "
      "collaborators at the <a href=\"http://dementia.ion.ucl.ac.uk/\">Dementia Research Centre</a>."
      "</p>"
      "<p>"
      "CMIC is a multi-disciplinary research group at University College London,  "
      "engaging in a wide array of medical imaging research. CMIC has produced several open source software packages"
      "available to the research community as separately downloadable packages, that can be run stand-alone or from within %1. These include:"
      "<ul>"
      "<li>NiftyReg: Fast linear and non-linear, multi-modal image registration for CPU/GPU.</li>"
      "<li>NiftySeg: Multi-channel image segmentation by Expectation Maximisation, with additional enhancements for cortical thickness estimation.</li>"
      "<li>NiftySim: Fast, non-linear elastic and visco-elastic finite element solver on CPU/GPU.</li>"
      "<li>NiftyRec: Fast stochastic emission tomography and multi-modal reconstruction on CPU/GPU.</li>"
      "</ul>"
      "For further details, go to <a href=\"http://cmic.cs.ucl.ac.uk/home/software/\">CMIC's software pages</a> or contact <a href=\"http://cmic.cs.ucl.ac.uk/staff/matt_clarkson/\">Matt Clarkson</a>."
      "</p>"
      "<p>"
      "NiftyView compiles and has been tested on MS&nbsp;Windows (XP and 7), "
      "Mac&nbsp;OS&nbsp;X (Leopard and Snow Leopard), and Linux (Centos and Ubuntu )"
      "</p>"
      ).arg(applicationName);


  QString TotalText = translatedTextAboutQtCaption.append(translatedTextAboutQtText);

  this->setWindowTitle(tr("About %1").arg(applicationName));
  QIcon helpAboutIcon(QLatin1String(":/images/icon.png"));

  if (!helpAboutIcon.isNull())
  {
    this->setWindowIcon(helpAboutIcon);
  }

  this->m_HelpAboutTextEdit->setText(TotalText); 

}

#endif
