\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{tubaf-thesis}[2024-04-18 TUBAF class for theses (based on tubaf-report) by Max Weiner]

\DeclareKeys[tubaf-thesis]{
    approved.if = @tubaf@thesis@approved,
    approved.usage = load,
}

\SetKeys[tubaf-thesis]{
    approved=false
}

\DeclareUnknownKeyHandler[tubaf-thesis]{%
    \PassOptionsToClass{\CurrentOption}{tubaf-report}
}

\ProcessKeyOptions[tubaf-thesis]

\LoadClass{tubaf-report}
\RequirePackage{scrbase}
\RequirePackage{tabularray}
\RequirePackage{xstring}

\providecaptionname{english}{\tubaf@thesis@supervisorsname}{\IfSubStr{\tubaf@thesis@supervisors}{\and}{Supervisors}{Supervisor}}
\providecaptionname{ngerman}{\tubaf@thesis@supervisorsname}{Betreuer}

\providecaptionname{english}{\tubaf@thesis@examinersname}{\IfSubStr{\tubaf@thesis@examiners}{\and}{Examiners}{Examiner}}
\providecaptionname{ngerman}{\tubaf@thesis@examinersname}{Prüfer}

\providecaptionname{english}{\tubaf@thesis@reviewersname}{\IfSubStr{\tubaf@thesis@reviewers}{\and}{Revierwers}{Reviewer}}
\providecaptionname{ngerman}{\tubaf@thesis@reviewersname}{Gutachter}

\providecaptionname{english}{\tubaf@thesis@coursename}{Course}
\providecaptionname{ngerman}{\tubaf@thesis@coursename}{Studiengang}

\providecaptionname{english}{\tubaf@thesis@submittedby}{submitted by}
\providecaptionname{ngerman}{\tubaf@thesis@submittedby}{vorgelegt von}

\providecaptionname{english}{\tubaf@thesis@bornon}{born on}
\providecaptionname{ngerman}{\tubaf@thesis@bornon}{geboren am}

\providecaptionname{english}{\tubaf@thesis@submittedto}{submitted to the}
\providecaptionname{ngerman}{\tubaf@thesis@submittedto}{eingereicht der}

\providecaptionname{english}{\tubaf@thesis@approvedby}{approved by the}
\providecaptionname{ngerman}{\tubaf@thesis@approvedby}{genehmigt durch die}

\providecaptionname{english}{\tubaf@thesis@toattain}{to attain the academic degree of}
\providecaptionname{ngerman}{\tubaf@thesis@toattain}{zur Erlangung des akademischen Grades}

\providecaptionname{english}{\tubaf@thesis@dateofsubmission}{Date of submission}
\providecaptionname{ngerman}{\tubaf@thesis@dateofsubmission}{Tag der Einreichung}

\providecaptionname{english}{\tubaf@thesis@dateofaward}{Date of award}
\providecaptionname{ngerman}{\tubaf@thesis@dateofaward}{Tag der Verleihung}

\providecaptionname{english}{\tubaf@thesis@matriculationnumbername}{Matriculation-Number}
\providecaptionname{ngerman}{\tubaf@thesis@matriculationnumbername}{Matrikelnummer}

\providecaptionname{english}{\tubaf@thesis@declarationofauthorshipname}{Declaration of Authorship}
\providecaptionname{ngerman}{\tubaf@thesis@declarationofauthorshipname}{Selbstständigkeitserklärung}

\providecaptionname{english}{\tubaf@thesis@declarationofauthorshiptext}{%
    I declare, that I have authored this work on my own under the guidance of my supervisors.
    I have not used any other tools than the declared.
    Parts of the work, which are taken from other works, if literally or in sense, I have cited correctly under specification of the source in each single case.
    This declaration also applies on images and figures.
    I declare, that this work satisfies the principles of academic integrity.
    I declare, that this work has not been presented to any other examination board, nor it will be.
}
\providecaptionname{ngerman}{\tubaf@thesis@declarationofauthorshiptext}{%
    Ich versichere, dass ich diese Arbeit selbstständig unter Anleitung meiner Betreuer verfasst habe.
    Ich habe keine anderen Hilfsmittel als die angegebenen benutzt.
    Die Stellen der Arbeit, die anderen Werken entnommen sind, ob dem Wortlaut oder dem Sinn nach, habe ich in jedem einzelnen Fall unter Angabe der Quelle als Entlehnung kenntlich gemacht.
    Diese Versicherung bezieht sich auch auf die bildlichen Darstellungen.
    Ich versichere, dass die Arbeit den Prinzipien der guten wissenschaftlichen Praxis genügt.
    Ich versichere, dass diese Arbeit bislang keiner anderen Prüfungskomission zur Begutachtung vorgelegt wurde noch wird.
}

\gdef\tubaf@thesis@supervisors{}
\newcommand{\supervisors}[1]{\gdef\tubaf@thesis@supervisors{#1}}

\gdef\tubaf@thesis@examiners{}
\newcommand{\examiners}[1]{\gdef\tubaf@thesis@examiners{#1}}

\gdef\tubaf@thesis@reviewers{}
\newcommand{\reviewers}[1]{\gdef\tubaf@thesis@reviewers{#1}}

\gdef\tubaf@thesis@course{}
\newcommand{\course}[1]{\gdef\tubaf@thesis@course{#1}}

\gdef\tubaf@thesis@faculty{}
\newcommand{\faculty}[1]{\gdef\tubaf@thesis@faculty{#1}}

\gdef\tubaf@thesis@currentdegree{}
\newcommand{\currentdegree}[1]{\gdef\tubaf@thesis@currentdegree{#1}}

\gdef\tubaf@thesis@attaineddegree{}
\gdef\tubaf@thesis@attaineddegree@short{}
\newcommand{\attaineddegree}[2][]{
    \gdef\tubaf@thesis@attaineddegree{#2}
    \IfArgIsEmpty{#1}{\gdef\tubaf@thesis@attaineddegree@short{#2}}{\gdef\tubaf@thesis@attaineddegree@short{#1}}
}

\gdef\tubaf@thesis@birthdate{}
\newcommand{\birthdate}[1]{\gdef\tubaf@thesis@birthdate{#1}}

\gdef\tubaf@thesis@birthplace{}
\newcommand{\birthplace}[1]{\gdef\tubaf@thesis@birthplace{#1}}

\gdef\tubaf@thesis@matriculationnumber{}
\newcommand{\matriculationnumber}[1]{\gdef\tubaf@thesis@matriculationnumber{#1}}

\newkomafont{faculty}{}
\newkomafont{attaineddegree}{\Large}
\setkomafont{author}{\Large}

\moretitle{
    \let\and\\
    \ifx\tubaf@thesis@faculty\empty\else
    \if@tubaf@thesis@approved
    \tubaf@thesis@approvedby
    \else
    \tubaf@thesis@submittedto
    \fi
    \\[5mm]
    {\usekomafont{faculty}\tubaf@thesis@faculty} \\[5mm]
    \fi
    \ifx\tubaf@thesis@attaineddegree\empty\else
    \tubaf@thesis@toattain \\[5mm]
    {\usekomafont{attaineddegree}\tubaf@thesis@attaineddegree \Ifstr{\tubaf@thesis@attaineddegree@short}{\tubaf@thesis@attaineddegree}{}{\ (\tubaf@thesis@attaineddegree@short)}} \\[5mm]
    \fi
    \tubaf@thesis@submittedby \\[5mm]
    {\usekomafont{author} \ifx\tubaf@thesis@currentdegree\empty\else\tubaf@thesis@currentdegree\ \fi \tubaf@thesis@author} \\[5mm]
    \ifx\tubaf@thesis@birthdate\empty\else\tubaf@thesis@bornon\ \tubaf@thesis@birthdate\ \ifx\tubaf@thesis@birthplace\empty\else in \tubaf@thesis@birthplace\fi\fi
}

\titlefoot{
    \let\and\\
    \edef\tubaf@thesis@foot@tablecontent{
        \ifx\tubaf@thesis@supervisors\empty\else
        \noexpand\tubaf@thesis@supervisorsname: & \tubaf@thesis@supervisors \\
        \fi
        \ifx\tubaf@thesis@examiners\empty\else
        \noexpand\tubaf@thesis@examinersname: & \tubaf@thesis@examiners \\
        \fi
        \if@tubaf@thesis@approved
        \ifx\tubaf@thesis@reviewers\empty\else
        \noexpand\tubaf@thesis@reviewersname: & \tubaf@thesis@reviewers \\
        \fi\fi
        \ifx\tubaf@thesis@course\empty\else
        \tubaf@thesis@coursename: & \tubaf@thesis@course \\
        \fi
        \ifx\tubaf@thesis@matriculationnumber\empty\else
        \tubaf@thesis@matriculationnumbername:    & \tubaf@thesis@matriculationnumber     \\
        \fi
        \if@tubaf@thesis@approved
        \tubaf@thesis@dateofaward: & \tubaf@thesis@date \\
        \else
        \tubaf@thesis@dateofsubmission: & \tubaf@thesis@date \\
        \fi
    }
    \begin{tblr}[expand=\tubaf@thesis@foot@tablecontent]{ll}
        \tubaf@thesis@foot@tablecontent
    \end{tblr}
}

\renewcommand{\tubaf@title@titlepage@prehook}{
    \let\tubaf@thesis@author\@author
    \let\tubaf@thesis@date\@date
    \let\@author\empty
    \let\@date\empty

    \renewcommand{\tubaf@title@titlepage@posthook}{
        \let\@author\tubaf@thesis@author
        \let\@date\tubaf@thesis@date
    }
}

\newcommand{\signaturefield}[2][7cm]{%
    \begin{minipage}[t]{#1}
        \raggedleft
        \rule{#1}{0.5pt}\\
        #2
    \end{minipage}
}

\NewDocumentCommand{\makedeclarationofauthorship}{O{\today} O{plain}}{%
    \chapter*{\tubaf@thesis@declarationofauthorshipname}
    \thispagestyle{#2}
    \tubaf@thesis@declarationofauthorshiptext

    \vspace{2cm}
    \noindent Freiberg, #1 \hfill \signaturefield{\tubaf@shortauthor}\\
    \cleardoublepage
}
