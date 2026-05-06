\ProvidesClass{diss}

\LoadClass[
    DIV=14,
    % DIV=14,
    listof=totoc,
    bibliography=totoc,
    twoside,
    open=right,
    BCOR=5mm,
    approved
]{tubaf-thesis}
\publishers{}

\usepackage[sans]{tubaf-fonts}
\usepackage[footmark=author-subject]{tubaf-marks}
\usepackage{tubaf-logoext-faculty-five}

\usepackage[ngerman,english]{babel}

\usepackage[mode=match, separate-uncertainty=true]{siunitx}

\usepackage{graphicx}
\usepackage{subcaption}
\renewcommand*{\figureformat}{\figurename~\thefigure}
\renewcommand*{\tableformat}{\tablename~\thetable}

\usepackage{tabularray}
\UseTblrLibrary{booktabs}
\UseTblrLibrary{varwidth}
\UseTblrLibrary{amsmath}

\usepackage{enumitem}

\usepackage[sorting=none,style=numeric,autolang=hyphen]{biblatex}
\addbibresource{refs.bib}
\setcounter{biburlucpenalty}{1000}
\setcounter{biburllcpenalty}{1000}

\usepackage{symbols}
\DeclareMathOperator{\sign}{sign}

\usepackage[a-2b]{pdfx}

\usepackage[hidelinks]{hyperref}
\usepackage[noabbrev]{cleveref}
\crefformat{pleq}{equations~(#2#1#3)}

\usepackage[record=nameref,abbreviations]{glossaries-extra}
\usepackage{glossaries}
\GlsXtrLoadResources[src=abbreviations]

\addtokomafont{chapterentry}{\normalfont\large}
\renewcommand\topfraction{0.9}
\renewcommand\bottomfraction{0.5}
\renewcommand\textfraction{0.1}
\renewcommand\floatpagefraction{0.85}

% \renewcommand\emph[1]{\textcolor{darkgray}{#1}}

\NewDocumentCommand{\mpunct}{m}{\,\text{#1}}
% OR
% \NewDocumentCommand{\mpunct}{m}{}

\usepackage{chemmacros}
