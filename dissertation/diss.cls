\ProvidesClass{diss}

\DeclareKeys{
    print.if=@diss@print,
    print.usage=load
}

\LoadClass[
    DIV=13,
    listof=totoc,
    bibliography=totoc,
]{tubaf-thesis}
\publishers{}

\if@diss@print
\SetKeys[tubaf-thesis]{twoside, open=right}
\setKeys[tubaf-color]{purpose=print}
\else
\SetKeys[tubaf-color]{purpose=digital}
\fi

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
\usepackage{symbols}
\DeclareMathOperator{\sign}{sign}

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

\usepackage{chemmacros}
