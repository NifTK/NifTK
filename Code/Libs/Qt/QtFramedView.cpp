/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-07-24 12:19:23 +0100 (Sun, 24 Jul 2011) $
 Revision          : $Revision: 6840 $
 Last modified by  : $Author: mjc $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
#ifndef QTFRAMEDVIEW_CPP
#define QTFRAMEDVIEW_CPP

#include "QtFramedView.h"
#include <QFrame>

QtFramedView::~QtFramedView()
{
}

QtFramedView::QtFramedView(QWidget *parent)
: QFrame(parent)
{
  QSizePolicy expandingSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
  expandingSizePolicy.setVerticalStretch(1);
  expandingSizePolicy.setHorizontalStretch(1);

  this->setSizePolicy(expandingSizePolicy);
  this->setAcceptDrops(true);
  this->SetFrameVisible(true);
  this->SetSelectedLook(false);
}

void QtFramedView::SetFrameToRed()
{
  this->setStyleSheet("QFrame {border: 2px solid red}");
}

void QtFramedView::SetFrameToWhite()
{
  this->setStyleSheet("QFrame {border: 2px solid white}");
}

void QtFramedView::SetFrameToBlack()
{
  this->setStyleSheet("QFrame {border: 2px solid black}");
}

void QtFramedView::SetFrameToYellow()
{
  this->setStyleSheet("QFrame {border: 2px solid yellow}");
}

void QtFramedView::SetFrameToGreen()
{
  this->setStyleSheet("QFrame {border: 3px solid green}");
}

void QtFramedView::SetFrameToBlue()
{
  this->setStyleSheet("QFrame {border: 3px solid blue}");
}

void QtFramedView::SetFrameVisible(bool b)
{
  if (b)
  {
    this->setFrameStyle(QFrame::Panel | QFrame::Plain);
    this->setContentsMargins(2, 2, 2, 2);
    this->setLineWidth(2);
    this->m_IsVisible = true;
  }
  else
  {
    this->setFrameStyle(QFrame::NoFrame);
    this->setContentsMargins(0, 0, 0, 0);
    this->setLineWidth(0);
    this->m_IsVisible = false;
  }
}

void QtFramedView::SetSelectedLook(bool b)
{
	if (b)
	{
	  this->SetFrameToRed();
		this->m_IsSelected = true;
	}
	else
	{
	  this->SetFrameToWhite();
		this->m_IsSelected = false;
	}
}

#endif
