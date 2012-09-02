#include "XnatFileNameTypeDialog.h"

#include <QComboBox>
#include <QDialogButtonBox>
#include <QFileInfo>
#include <QGridLayout>
#include <QLabel>
#include <QLineEdit>

XnatFileNameTypeDialog::XnatFileNameTypeDialog(const QString& filters, QWidget* parent)
: QDialog(parent)
{
  QLabel* nameLabel = new QLabel(tr("File name:"));
  nameEdit = new QLineEdit;

  QLabel* typeLabel = new QLabel(tr("Files of type:"));
  typeComboBox = new QComboBox;

  filterList = makeFilterList(filters);
  if ( filterList.isEmpty() )
  {
    filterList << "All Files (*)";
  }
  typeComboBox->insertItems(0, filterList);
  typeComboBox->setCurrentIndex(0);   //  necessary?????

  QDialogButtonBox* buttonBox = new QDialogButtonBox(QDialogButtonBox::Ok|
                                                     QDialogButtonBox::Cancel);
  connect(buttonBox, SIGNAL(accepted()), this, SLOT(accept()));
  connect(buttonBox, SIGNAL(rejected()), this, SLOT(reject()));

  QGridLayout* gridLayout = new QGridLayout;
  gridLayout->addWidget(nameLabel, 0, 0);
  gridLayout->addWidget(nameEdit, 0, 1);
  gridLayout->addWidget(typeLabel, 1, 0);
  gridLayout->addWidget(typeComboBox, 1, 1);
  gridLayout->addWidget(buttonBox, 2, 0, 1, 2);
  setLayout(gridLayout);

  setWindowTitle(tr("Save File to XNAT"));
  setFixedHeight(sizeHint().height());
  setMinimumWidth(400);
}

void XnatFileNameTypeDialog::accept()
{
  // get input filename
  QString fname = nameEdit->text();
  if ( fname.isEmpty() || ( fname == "." ) )
  {
    // QMessageBox::warning(this, tr("Missing File Name Error"), tr("Please enter file name."));
    // nameEdit->selectAll();
    // nameEdit->setFocus();
    nameEdit->clear();
    return;
  }

  // get selected file type
  QString ftype = typeComboBox->currentText();

  // append selected file type, if necessary
  inputFname = fixFilePathExtension(fname, ftype);

  QDialog::accept();
}

QStringList XnatFileNameTypeDialog::makeFilterList(const QString& filters)
{
  // check if no file extensions input
  if ( filters.isEmpty() )
  {
    return QStringList();
  }

  // determine if separator for file extensions is ;; or \n
  //   (default is ;; if no separators present)
  QString sep(";;");
  if ( filters.indexOf(sep, 0)  == -1 )
  {
    if ( filters.indexOf("\n", 0) != -1 )
    {
        sep = "\n";
    }
  }

  // generate list of file extensions
  return filters.split(sep, QString::SkipEmptyParts);
}

QStringList XnatFileNameTypeDialog::getWildCardListFromFilter(const QString& filter)
{
  QString wildCardExts;

  // extract wild cards from file extension filter
  // if we have (...) in our filter, strip everything out but the contents of ()
  int start = filter.indexOf('(');
  int end   = filter.lastIndexOf(')');
  if( ( start != -1 ) && ( end != -1 ) )
  {
    wildCardExts = filter.mid(start + 1, end - start - 1);
  }

  // generate list of wild cards -- wild cards separated by spaces or semi-colons
  QStringList wildCardExtList = wildCardExts.split(QRegExp("[\\s+;]"), QString::SkipEmptyParts);

  // add a *.ext.* for every *.ext in order to support file groups
  QStringList wildCardList = wildCardExtList;
  foreach( QString ext, wildCardExtList )
  {
    wildCardList.append(ext + ".*");
  }

  return wildCardList;
}

QString XnatFileNameTypeDialog::fixFilePathExtension(const QString& filePath, const QString& filter)
{
  // get extension from input filename
  QFileInfo fileInfo(filePath);
  QString ext = fileInfo.completeSuffix();

  // check that extension added by user is one of supported types
  if ( !ext.isEmpty() )
  {
    bool pass = false;
    QString fname = fileInfo.fileName();
    foreach ( QString currFilter, filterList )
    {
      QStringList wildCardList = getWildCardListFromFilter(currFilter);

      foreach ( QString wildCard, wildCardList )
      {
        if ( wildCard.indexOf('.') != -1 )
        {
          // validate extension, not filename
          wildCard = QString("*.%1").arg(wildCard.mid(wildCard.indexOf('.') + 1));
          QRegExp regEx = QRegExp(wildCard, Qt::CaseInsensitive, QRegExp::Wildcard);
          if ( regEx.exactMatch(fname) )
          {
            pass = true;
            break;
          }
        }
        else
        {
          // filter does not specify any rule for extension
          // assume input extension is matched
          pass = true;
          break;
        }
      }
      if ( pass )
      {
        break;
      }
    }
    if ( !pass )
    {
        // force adding of selected extension
        ext = QString();
    }
  }

  // initilize output for filename with valid extension
  QString fixedFilePath = filePath;

  // append selected extension if input filename does not end with valid extension
  if ( ext.isEmpty() )
  {
    // get selected extension
    QString extensionWildCard = getWildCardListFromFilter(filter).first();
    QString selectedExtension = extensionWildCard.mid(extensionWildCard.indexOf('.') + 1);

    // append selected extension if it exists
    if ( !selectedExtension.isEmpty() && selectedExtension != "*" )
    {
      if ( fixedFilePath.at(fixedFilePath.size() - 1) != '.' )
      {
        fixedFilePath += ".";
      }
      fixedFilePath += selectedExtension;
    }
  }

  return fixedFilePath;
}
