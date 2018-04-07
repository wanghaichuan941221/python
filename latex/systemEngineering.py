from pylatex import Document,Section,Subsection,Command
from pylatex.utils import italic,NoEscape

def fill_document(doc):
    with doc.create(Section('A Section')):
        doc.append('tex')
        doc.append(italic('italic tex'))

        with doc.create(Subsection('A Subsection')):
            doc.append('some tex')


if __name__ == '__main__':
    doc=Document('basic')
    fill_document(doc)

    doc.generate_pdf(clean_tex=False)
    #document with make title
    doc=Document()
    doc.preamble.append(Command('title','system engineering'))
    doc.preamble.append(Command('author','the author'))
    doc.preamble.append(Command('date',NoEscape(r'\today')))
    doc.append(NoEscape(r'\maketitle'))
    fill_document(doc)
    
    doc.generate_pdf('system',clean_tex=False)
