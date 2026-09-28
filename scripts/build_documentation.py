"""Build the dated technical handbook from reviewed Markdown and research history.

Install the project's docs extra. No network, training or model inference occurs.
"""
from __future__ import annotations
import hashlib
import html
import json
from pathlib import Path
import re
import textwrap
import markdown
from bs4 import BeautifulSoup, NavigableString, Tag
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import BaseDocTemplate, Frame, PageTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle, Image, KeepTogether, Flowable
from reportlab.platypus.tableofcontents import TableOfContents

ROOT=Path(__file__).resolve().parents[1]
DOCS=ROOT/'docs'
OUTPUT=DOCS/'Monocular_Surface_Reconstruction_Technical_Documentation.pdf'
CHAPTERS=['OVERVIEW','USER_GUIDE','ARCHITECTURE','SCIENTIFIC_CONTRACTS','METHODS','CODE_MAP','DATA','VALIDATION','RESULTS','EXPERIMENTS','DECISIONS','FEATURES','MODEL_CARDS','API','SETUP','PERFORMANCE','SECURITY_REVIEW','RELEASE','ROADMAP','GLOSSARY','REFERENCES','SOURCE_INVENTORY']
WIDTH,HEIGHT=A4
MARGIN=48
CONTENT=WIDTH-2*MARGIN
INK=colors.HexColor('#18313d');TEAL=colors.HexColor('#087f83');MUTED=colors.HexColor('#526975');LIGHT=colors.HexColor('#edf4f5')

def fonts():
    candidates=[(Path('C:/Windows/Fonts'),'calibri.ttf','calibrib.ttf','calibrii.ttf'),(Path('/usr/share/fonts/truetype/dejavu'),'DejaVuSans.ttf','DejaVuSans-Bold.ttf','DejaVuSans-Oblique.ttf')]
    for base,reg,bold,italic in candidates:
        if (base/reg).exists():
            for name,file in [('Body',reg),('Body-Bold',bold),('Body-Italic',italic)]:pdfmetrics.registerFont(TTFont(name,str(base/file)))
            pdfmetrics.registerFontFamily('Body',normal='Body',bold='Body-Bold',italic='Body-Italic',boldItalic='Body-Bold')
            return 'Body'
    return 'Helvetica'

FONT=fonts()
ST=getSampleStyleSheet()
ST.add(ParagraphStyle('Text',fontName=FONT,fontSize=9.5,leading=14.1,textColor=INK,spaceAfter=7,splitLongWords=True,allowWidows=0,allowOrphans=0))
ST.add(ParagraphStyle('SmallText',parent=ST['Text'],fontSize=8,leading=11,textColor=MUTED))
ST.add(ParagraphStyle('Cell',parent=ST['Text'],fontSize=7.6,leading=10.4,spaceAfter=0))
ST.add(ParagraphStyle('CodeCell',parent=ST['Text'],fontName='Courier',fontSize=7,leading=10,backColor=LIGHT,borderPadding=7,spaceBefore=4,spaceAfter=10))
ST.add(ParagraphStyle('Chapter',parent=ST['Text'],fontSize=23,leading=28,textColor=INK,spaceBefore=9,spaceAfter=19,keepWithNext=True))
ST.add(ParagraphStyle('Sub',parent=ST['Text'],fontSize=13,leading=17,textColor=TEAL,spaceBefore=12,spaceAfter=7,keepWithNext=True))
ST.add(ParagraphStyle('Minor',parent=ST['Text'],fontSize=10.5,leading=14,textColor=INK,spaceBefore=10,spaceAfter=5,keepWithNext=True))

class Cover(Flowable):
    def __init__(self):Flowable.__init__(self);self.width=CONTENT;self.height=HEIGHT-114
    def draw(self):
        c=self.canv;w=self.width;h=self.height
        c.setFillColor(INK);c.rect(-MARGIN,-60,WIDTH,HEIGHT,stroke=0,fill=1)
        c.setFillColor(colors.HexColor('#4bdcc9'));c.setFont(FONT,10);c.drawString(0,h-25,'RESEARCH SYSTEMS  /  TECHNICAL HANDBOOK')
        c.setFillColor(colors.white);c.setFont(FONT,40)
        for i,line in enumerate(['Monocular','Surface','Reconstruction']):c.drawString(0,h-104-i*47,line)
        c.setFont(FONT,13);c.setFillColor(colors.HexColor('#c4dddf'));c.drawString(0,h-260,'Single-image geometry, metric height and semantic analysis')
        c.setFillColor(colors.HexColor('#4bdcc9'));c.setFont(FONT,14);c.drawString(0,h-310,'Technical Documentation — Status as of 28 September 2026')
        c.setFont(FONT,10);c.setFillColor(colors.white);c.drawString(0,h-334,'Version 0.1.0   •   Active research prototype')
        # Analytic vector surface, not a model result.
        import math
        c.setStrokeColor(colors.HexColor('#368f92'));c.setLineWidth(.55)
        for j in range(18):
            pts=[]
            for i in range(42):
                x=i*10.5+j*4;y=62+j*5+26*math.sin(i*.17+j*.25)+52*math.exp(-((i-21)/9)**2-((j-9)/7)**2)
                pts.append((x,y))
            p=c.beginPath();p.moveTo(*pts[0])
            for x,y in pts[1:]:p.lineTo(x,y)
            c.drawPath(p)
        c.setFillColor(colors.HexColor('#c4dddf'));c.setFont(FONT,9)
        c.drawString(0,18,'Methods • Architecture • Evidence • Decisions • Reproducibility')
        c.drawString(0,0,'github.com/GauravR17A/monocular-surface-reconstruction')

class Handbook(BaseDocTemplate):
    def __init__(self,filename):
        super().__init__(str(filename),pagesize=A4,leftMargin=MARGIN,rightMargin=MARGIN,topMargin=49,bottomMargin=48,title='Monocular Surface Reconstruction — Technical Documentation — 28 September 2026',author='Monocular Surface Reconstruction project',subject='Research prototype: methods, implementation, decisions and measured evidence',pageCompression=1)
        self.addPageTemplates(PageTemplate(id='main',frames=[Frame(MARGIN,48,CONTENT,HEIGHT-97,id='normal',leftPadding=0,rightPadding=0,topPadding=0,bottomPadding=0)],onPage=self.decorate))
        self.heading='Technical handbook'
    def beforeDocument(self):self.heading='Technical handbook'
    def decorate(self,c,doc):
        if doc.page==1:return
        c.saveState();c.setStrokeColor(colors.HexColor('#d5e2e5'));c.setLineWidth(.5);c.line(MARGIN,HEIGHT-33,WIDTH-MARGIN,HEIGHT-33)
        c.setFont(FONT,7.3);c.setFillColor(MUTED);c.drawString(MARGIN,HEIGHT-25,'MONOCULAR SURFACE RECONSTRUCTION  /  TECHNICAL DOCUMENTATION')
        c.drawString(MARGIN,28,'STATUS AS OF 28 SEPTEMBER 2026  •  RESEARCH PROTOTYPE')
        c.drawRightString(WIDTH-MARGIN,28,str(doc.page));c.restoreState()
    def afterFlowable(self,flow):
        if isinstance(flow,Paragraph) and getattr(flow,'chapter_key',None):
            title=flow.getPlainText();key=flow.chapter_key
            self.canv.bookmarkPage(key);self.canv.addOutlineEntry(title,key,0,False)
            self.notify('TOCEntry',(0,title,self.page,key))

def inline(node):
    if isinstance(node,NavigableString):return html.escape(str(node))
    if not isinstance(node,Tag):return ''
    inside=''.join(inline(c) for c in node.children)
    if node.name in ('strong','b'):return '<b>'+inside+'</b>'
    if node.name in ('em','i'):return '<i>'+inside+'</i>'
    if node.name=='code':return '<font name="Courier" size="8">'+inside+'</font>'
    if node.name=='a':
        link=node.get('href','')
        if link.startswith(('https://','http://')):return '<a href="'+html.escape(link,quote=True)+'" color="#087f83">'+inside+'</a>'
        return '<font color="#087f83">'+inside+'</font>'
    if node.name=='br':return '<br/>'
    if node.name=='img':return ''
    return inside

def chapter(file,index):
    result=[]
    source=file.read_text(encoding='utf-8')
    soup=BeautifulSoup(markdown.markdown(source,extensions=['tables','fenced_code','sane_lists']),'html.parser')
    first=True
    for el in soup.children:
        if not isinstance(el,Tag):continue
        tag=el.name
        if tag in ('h1','h2','h3','h4','h5','h6'):
            if tag=='h1' and first:
                title=f'{index:02d}  {el.get_text(" ")}'
                p=Paragraph(html.escape(title),ST['Chapter']);p.chapter_key=f'chapter-{index}'
                result.append(p);first=False
                result.append(Paragraph('SOURCE  '+html.escape(file.relative_to(ROOT).as_posix()),ST['SmallText']))
            else:result.append(Paragraph(inline(el),ST['Sub'] if tag in ('h1','h2') else ST['Minor']))
        elif tag=='table':
            rows=[]
            for tr in el.find_all('tr'):
                row=[]
                for td in tr.find_all(['th','td'],recursive=False):
                    content=inline(td)
                    row.append(Paragraph(('<b>'+content+'</b>') if td.name=='th' else content,ST['Cell']))
                if row:rows.append(row)
            if not rows:continue
            n=max(map(len,rows))
            for row in rows:row.extend([Paragraph('',ST['Cell'])]*(n-len(row)))
            # Content-aware widths with bounded imbalance for readable wide tables.
            scores=[]
            for c in range(n):
                lengths=sorted(len(row[c].getPlainText()) for row in rows)
                scores.append(max(8,min(65,lengths[int((len(lengths)-1)*.75)]))**.55)
            widths=[CONTENT*s/sum(scores) for s in scores]
            t=Table(rows,colWidths=widths,repeatRows=1,hAlign='LEFT',splitByRow=1)
            t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),LIGHT),('VALIGN',(0,0),(-1,-1),'TOP'),('LINEBELOW',(0,0),(-1,0),.8,TEAL),('LINEBELOW',(0,1),(-1,-1),.25,colors.HexColor('#dce6e9')),('LEFTPADDING',(0,0),(-1,-1),5),('RIGHTPADDING',(0,0),(-1,-1),5),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6)]))
            result.extend([t,Spacer(1,10)])
        elif tag=='pre':
            text=el.get_text().rstrip()
            if text.startswith('graph ') or text.startswith('flowchart '):
                text='Diagram source (rendered architecture is provided in the current architecture chapter).\n'+text
            lines=[]
            for line in text.splitlines():lines.extend(textwrap.wrap(line.expandtabs(4),width=91,replace_whitespace=False,drop_whitespace=False) or [' '])
            # Break very long code blocks into page-safe groups without losing source.
            for start in range(0,len(lines),34):
                encoded='<br/>'.join(html.escape(x).replace(' ','&#160;') for x in lines[start:start+34])
                result.append(Paragraph(encoded,ST['CodeCell']))
        elif tag in ('ul','ol'):
            for i,li in enumerate(el.find_all('li',recursive=False),1):
                prefix=f'{i}. ' if tag=='ol' else '• '
                result.append(Paragraph(html.escape(prefix)+inline(li),ST['Text']))
        elif tag=='hr':result.append(Spacer(1,8))
        else:
            for img in el.find_all('img'):
                asset=(file.parent/img.get('src','')).resolve()
                if asset.suffix.lower() in ('.png','.jpg','.jpeg') and asset.exists():
                    from PIL import Image as PILImage
                    iw,ih=PILImage.open(asset).size;w=CONTENT;h=w*ih/iw
                    if h>365:w*=365/h;h=365
                    result.append(Image(str(asset),width=w,height=h,hAlign='LEFT'))
                    if img.get('alt'):result.append(Paragraph(html.escape(img['alt']),ST['SmallText']))
            txt=inline(el).strip()
            if txt:result.append(Paragraph(txt,ST['Text']))
    return result

def main():
    sources=[DOCS/(name+'.md') for name in CHAPTERS]+[ROOT/'THIRD_PARTY_NOTICES.md']+sorted((DOCS/'history').glob('*.md'))
    missing=[str(p) for p in sources if not p.exists()]
    if missing:raise FileNotFoundError('\n'.join(missing))
    story=[Cover(),PageBreak(),Paragraph('How to use this handbook',ST['Chapter']),Paragraph('This is a dated technical snapshot of an active research prototype. Current chapters explain the implemented system, scientific boundaries and available evidence. The historical section preserves protocols, outcomes and decisions that led to that state. A failed experiment is retained as evidence; a planned feature is not presented as complete.',ST['Text']),Paragraph('Read Overview and User guide for the system; Scientific contracts and Validation for interpretation; Results, Experiments and Decisions for what was learned; Setup and Release for reproduction. The source repository is the editable companion to this PDF.',ST['Text']),Paragraph('Historical records use normalised public names and paths. Exact old source hashes describe their original archive, not renamed files. Full private data/run archives are not bundled. See the source inventory and evidence directory for the included records and verification scope.',ST['Text']),Paragraph('Publication identity: version 0.1.0 • 28 September 2026. No blanket open-source or commercial-use licence is granted; third-party notices apply.',ST['SmallText']),PageBreak(),Paragraph('Contents',ST['Chapter'])]
    toc=TableOfContents();toc.levelStyles=[ParagraphStyle('ContentsEntry',fontName=FONT,fontSize=9,leading=13.5,textColor=INK,spaceBefore=3,leftIndent=0,firstLineIndent=0)]
    story.extend([toc,PageBreak()])
    for i,file in enumerate(sources,1):
        if i>1:story.append(PageBreak())
        story.extend(chapter(file,i))
    doc=Handbook(OUTPUT);doc.multiBuild(story)
    result={'status_date':'2026-09-28','file':OUTPUT.relative_to(ROOT).as_posix(),'chapters':len(sources),'sha256':hashlib.sha256(OUTPUT.read_bytes()).hexdigest(),'bytes':OUTPUT.stat().st_size,'sources':[{'path':p.relative_to(ROOT).as_posix(),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sources]}
    try:
        import fitz
        with fitz.open(OUTPUT) as pdf:result.update(pages=len(pdf),bookmarks=len(pdf.get_toc()))
    except ImportError:pass
    (DOCS/'evidence/documentation-build.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='sources'},indent=2))

if __name__=='__main__':main()
