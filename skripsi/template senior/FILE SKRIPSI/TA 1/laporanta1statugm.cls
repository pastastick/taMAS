%Created by Solikhatun, Sekretaris prodi Ilmu Aktuaria FMIPA UGM
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{laporanta1statugm}[2018/05/09 v1.0 The Universitas Gadjah Mada Mathematics Thesis Report Class]

\LoadClass[a4paper,12pt]{report}
\RequirePackage[english,bahasa]{babel}
\RequirePackage{times}
\RequirePackage{setspace}
\RequirePackage{indentfirst}

\RequirePackage{graphicx,latexsym}
\RequirePackage{amssymb}
\RequirePackage{amsfonts}
\RequirePackage{amstext}
\RequirePackage{amsmath}
\RequirePackage{amsthm}
\RequirePackage{listings}
\RequirePackage{color}
\RequirePackage{float}
\RequirePackage{longtable}
\RequirePackage{enumerate}
\RequirePackage{multirow}
\RequirePackage[center,font=footnotesize,labelfont=bf,textfont=bf,labelsep=space,aboveskip=1.5ex,singlelinecheck=off]{caption}
\RequirePackage{pdfpages}
\RequirePackage{tikz}
\RequirePackage{cases}
\usetikzlibrary{matrix,arrows}

\RequirePackage{natbib}
\bibliographystyle{plainnat}

\renewcommand{\baselinestretch}{1.5}
\parskip0.15cm

\renewcommand{\normalsize}{\fontsize{12bp}{14bp}\selectfont}
\frenchspacing

%------------------------------------------------------------
%Layout
%------------------------------------------------------------
\setlength{\topmargin}{0cm}
\setlength{\headheight}{0.46cm}
\setlength{\headsep}{1cm}
\setlength{\topskip}{0cm}
\setlength{\oddsidemargin}{1.46cm}
\setlength{\evensidemargin}{1.46cm}
\setlength{\textwidth}{14cm}
\setlength{\textheight}{22.7cm}
\setlength{\footskip}{1.3cm}
\setlength{\marginparsep}{0pt}
\setlength{\marginparwidth}{0pt}
\setlength{\parindent}{3em}

\pagestyle{myheadings}


\setcounter{secnumdepth}{2}
\renewcommand{\thechapter}{\Roman{chapter}}
\renewcommand{\thesection}{\arabic{chapter}.\arabic{section}.}
\renewcommand{\thesubsection}{\arabic{chapter}.\arabic{section}.\arabic{subsection}.}
%\renewcommand{\thesubsubsection}{\arabic{chapter}.\arabic{section}.\arabic{subsection}.\arabic{subsubsection}.}

\renewcommand{\thetable}{\arabic{chapter}.\arabic{table}}
\renewcommand{\thefigure}{\arabic{chapter}.\arabic{figure}}
\renewcommand{\theequation}{\arabic{chapter}.\arabic{equation}}

\usepackage{titlesec}
\titleformat{\chapter}
%[hang]
%[runin]
%[leftmargin]
[display]
{\center\large\bfseries}
{\large\MakeUppercase{\chaptertitlename} \thechapter}{1ex}{\large}
\titlespacing{\chapter}{0cm}{-1.2cm}{2cm}

\titleformat{\section}
[hang]
{\bfseries}
{\bfseries\thesection}{1ex}{\bfseries}
\titlespacing{\section}{0cm}{3ex}{0ex}


\titleformat{\subsection}
[hang]
{\bfseries}
{\bfseries\thesubsection}{1ex}{\bfseries}
\titlespacing{\subsection}{0cm}{3ex}{0ex}



\usepackage{titletoc}
\titlecontents{chapter}[0cm]
  {\vspace{-1.3ex}}
  {\bfseries\parbox{1.75cm}{\vspace{-0.5ex}\chaptername\ \thecontentslabel}}
  {\bfseries}
  {\bfseries\titlerule*[0.2cm]{.}\contentspage}

\titlecontents{section}[1.8em]
  {\vspace{-1.7ex}}
  {\hspace{-1.75em}\thecontentslabel ~~}
  {}
  {\titlerule*[0.2cm]{.}\contentspage}

\titlecontents{subsection}[1.9cm]
  {\vspace{-1.7ex}}
  {\hspace{-2.55em}\thecontentslabel ~~}
  {}
  {\titlerule*[0.2cm]{.}\contentspage}


%Untuk setting penomoran teorema, lemma, definisi, dll..
\newtheoremstyle{dotlesstheorem}  % follow `plain` defaults but change HEADSPACE.
  {}   % ABOVESPACE
  {0.4cm}   % BELOWSPACE
  {\itshape}  % BODYFONT
  {0pt}       % INDENT (empty value is the same as 0pt)
  {\bfseries} % HEADFONT
  {}         % HEADPUNCT
  {  }  % HEADSPACE. `plain` default: {5pt plus 1pt minus 1pt}
  {}          % CUSTOM-HEAD-SPEC
\theoremstyle{dotlesstheorem}
\newtheorem{teorema}{Teorema}[section]
\renewcommand{\theteorema}{\arabic{chapter}.\arabic{section}.\arabic{teorema}}
\newtheorem{lemma}[teorema]{Lema}
\newtheorem{sifat}[teorema]{Sifat}
\newtheorem{akibat}[teorema]{Akibat}
\newtheorem{proposisi}[teorema]{Proposisi}
\newtheorem{definisi}[teorema]{Definisi}

\newtheoremstyle{dotlessexample}  % follow `plain` defaults but change HEADSPACE.
  {}   % ABOVESPACE
  {0.4cm}   % BELOWSPACE
  {}  % BODYFONT
  {0pt}       % INDENT (empty value is the same as 0pt)
  {\bfseries} % HEADFONT
  {}         % HEADPUNCT
  {  }  % HEADSPACE. `plain` default: {5pt plus 1pt minus 1pt}
  {}          % CUSTOM-HEAD-SPEC
\theoremstyle{dotlessexample}
\newtheorem{contoh}[teorema]{Contoh}
\newtheorem{algoritma}{Algoritma}[chapter]
\renewcommand{\thealgoritma}{\arabic{chapter}.\arabic{algoritma}}

\renewenvironment{proof}{\vspace{-2ex}\noindent{\bf Bukti.}}{\hfill$\blacksquare$\newline\vspace{-0.4cm}}
\def\faktor#1#2{\raise0.7ex\hbox{$#1$}\!\!\!\rotatebox{-11}{$\Big/$\!}\!\lower0.7ex\hbox{$#2$}}
\def\faktorr#1#2{\raise0.4ex\hbox{$#1$}\!\!\rotatebox{-11}{$\big/$\!}\lower0.4ex\hbox{$#2$}}
\def\N{\mathbb{N}}
\def\Z{\mathbb{Z}}
\def\Q{\mathbb{Q}}
\def\R{\mathbb{R}}
\def\C{\mathbb{C}}
\def\im{\textrm{Im}}
\def\pustaka#1{\emph{(\citeauthor{#1},~~\!\citeyear{#1})}}
\def\tpustaka#1{\citeauthor{#1} (\citeyear{#1})}
\def\itemset{\vspace{-1ex}\itemsep3pt \parskip0pt \parsep0pt}

\usepackage[pdftex,bookmarks=true]{hyperref}


\def\captionsbahasa{%
\def\prefacename{PRAKATA}%
\def\contentsname{DAFTAR ISI}%
\def\listfigurename{DAFTAR GAMBAR}%
\def\listtablename{DAFTAR TABEL}%
\def\listappendixname{DAFTAR LAMPIRAN}%
\def\nomenclaturename{DAFTAR LAMBANG}%
\def\abstractname{INTISARI}%
\def\partname{BAGIAN}%
\def\chaptername{BAB}%
\def\appendixname{LAMPIRAN}%
\def\refname{DAFTAR PUSTAKA}%
\def\bibname{DAFTAR PUSTAKA}%
\def\indexname{Indek}%
\def\figurename{Gambar}%
\def\tablename{Tabel}%
\def\pagename{Halaman}%
}
\def\captionsenglish{%
\def\prefacename{PREFACE}%
\def\contentsname{CONTENTS}%
\def\listfigurename{LIST OF FIGURES}%
\def\listtablename{LIST OF TABLES}%
\def\listappendixname{LIST OF APPENDICES}%
\def\nomenclaturename{NOMENCLATURE}%
\def\abstractname{ABSTRACT}%
\def\partname{PART}%
\def\chaptername{CHAPTER}%
\def\appendixname{APPENDIX}%
\def\refname{REFERENCES}%
\def\bibname{REFERENCES}%
\def\indexname{Index}%
\def\figurename{Figure}%
\def\tablename{Table}%
\def\pagename{Page}%
}

%------------------------------------------------------------
%The form of page numbering
%------------------------------------------------------------
\pagestyle{myheadings}
\markright{}

\parskip0.12cm

%------------------------------------------------------------
%Define page numbering in the first chapter
%------------------------------------------------------------
\def\ps@chapterheading{%
  \let\@evenhead\@empty\let\@oddhead\@empty
  \def\@oddfoot{\hfil\thepage\hfil}%
  \def\@evenfoot{\hfil\thepage\hfil}
  }


%------------------------------------------------------------]
%Define thesis's inputs
%------------------------------------------------------------
\gdef\@university{Universitas Gadjah Mada}
\gdef\@faculty{Fakultas Matematika dan Ilmu Pengetahuan Alam}
\gdef\@dept{Matematika}
\gdef\@program{Statistika}
\gdef\@city{Yogyakarta}
\newcommand{\titleind}[1]{\gdef\@titleind{#1}}
\newcommand{\@titleind}{}
\newcommand{\fullname}[1]{\gdef\@fullname{#1}}
\newcommand{\@fullname}{}
\newcommand{\NIM}[1]{\gdef\@NIM{#1}}
\newcommand{\@NIM}{}
\newcommand{\yearsubmit}[1]{\gdef\@yearsubmit{#1}}
\newcommand{\@yearsubmit}{}
\newcommand{\firstsupervisor}[1]{\gdef\@firstsupervisor{#1}}
\newcommand{\@firstsupervisor}{}
\newcommand{\secondsupervisor}[1]{\gdef\@secondsupervisor{#1}}
\newcommand{\@secondsupervisor}{}


%------------------------------------------------------------
%Define cover in Indonesian
%------------------------------------------------------------
\def\cover{%
  \thispagestyle{empty}%
  \pagenumbering{roman}
  \setcounter{page}{1}
    \begin{center}
      \begin{singlespace}
      \MakeUppercase{\normalfont\bfseries Laporan Tugas Akhir I}\\
      \vspace{0.75cm}	
      \MakeUppercase{\large\bfseries\@titleind}\par\nobreak
       \vfill
       \includegraphics[width=5.5cm]{logougm.png}
       \vfill
       {\normalfont
        \@fullname\\
        \@NIM}\\
       \vspace{1cm}

       {\normalfont
       \begin{tabular}{rl}
       PEMBIMBING~:&\@firstsupervisor\\
       &\@secondsupervisor\\
       \end{tabular}
       }

       \vspace{1.5cm}
       {\normalfont
         \MakeUppercase{\normalfont\bfseries Program Sarjana Program Studi \@program}\\
         \MakeUppercase{\normalfont\bfseries Departemen \@dept}\\
         \MakeUppercase{\normalfont\bfseries\@faculty}\\
         \MakeUppercase{\normalfont\bfseries\@university}\\
         \MakeUppercase{\normalfont\bfseries\@city}\\
         \vspace{0.3cm}
         {\normalfont\bfseries\@yearsubmit}}\\
      \end{singlespace}
      \end{center}
      \clearpage\pagenumbering{roman}\setcounter{page}{0}
}

\endinput
