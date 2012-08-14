#include "XnatNameDialog.h"

#include <QDialogButtonBox>
#include <QLabel>
#include <QLineEdit>
#include <QMessageBox>
#include <QRegExp>
#include <QRegExpValidator>
#include <QVBoxLayout>

XnatNameDialog::XnatNameDialog(QWidget* p, const QString& kind, const QString& parentName)
: QDialog(p)
{
  QString childKind(kind.toLower());
  QLabel* nameLabel = new QLabel(tr("Name of new %1 in %2:").arg(childKind).arg(parentName));

  nameEdit = new QLineEdit;
  QRegExp regExp("[A-Za-z0-9][A-Za-z0-9_]*");
  nameEdit->setValidator(new QRegExpValidator(regExp, this));

  QDialogButtonBox* buttonBox = new QDialogButtonBox(QDialogButtonBox::Ok|
                                                     QDialogButtonBox::Cancel);
  connect(buttonBox, SIGNAL(accepted()), this, SLOT(accept()));
  connect(buttonBox, SIGNAL(rejected()), this, SLOT(reject()));

  QVBoxLayout* layout = new QVBoxLayout;
  layout->addWidget(nameLabel);
  layout->addWidget(nameEdit);
  layout->addWidget(buttonBox);
  setLayout(layout);

  if ( childKind.size() > 0 )
  {
    childKind[0] = childKind[0].toUpper();
  }
  setWindowTitle(tr("Create New %1").arg(childKind));
  setFixedHeight(sizeHint().height());
  setMinimumWidth(320);
  setModal(true);
}

const QString XnatNameDialog::getNewName()
{
  return newName;
}

void XnatNameDialog::accept()
{
  newName = nameEdit->text();
  if ( newName.isEmpty() )
  {
    QMessageBox::warning(this, tr("Missing Name Error"), tr("Please enter name or cancel."));
    nameEdit->selectAll();
    nameEdit->setFocus();
    return;
  }

  QDialog::accept();
}
