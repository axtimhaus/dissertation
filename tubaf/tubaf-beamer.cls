\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{tubaf-beamer}[2024-04-18 TUBAF class for presentation slides (based on beamer) by Max Weiner]

\DeclareKeys[tubaf-beamer]{%
    title-dark.if = @tubaf@beamer@title@dark,
    title-dark.usage = preamble,
    title-graphic-opacity.store = \tubaf@beamer@title@graphic@opacity,
    title-graphic-opacity.usage = preamble,
    foot-trapezoid.if = @tubaf@beamer@foot@trapezoid,
    foot-trapezoid.usage = preamble,
    foot-trapezoid-opacity.store = \tubaf@beamer@foot@trapezoid@opacity,
    foot-trapezoid-opacity.usage = preamble,
    color-primary.store = \tubaf@beamer@color@primary,
    color-primary.usage = preamble,
    color-secondary.store = \tubaf@beamer@color@secondary,
    color-secondary.usage = preamble,
    color-tertiary.store = \tubaf@beamer@color@tertiary,
    color-tertiary.usage = preamble,
}

\SetKeys[tubaf-beamer]{
    title-dark=false,
    title-graphic-opacity=0.6,
    foot-trapezoid=false,
    foot-trapezoid-opacity=0.15,
    color-primary=tubaf-primary,
    color-secondary=tubaf-accent-red6,
    color-tertiary=tubaf-accent-blue1,
}

\DeclareUnknownKeyHandler[tubaf-beamer]{%
    \PassOptionsToClass{\CurrentOption}{beamer}
}

\ProcessKeyOptions[tubaf-beamer]

\LoadClass[aspectratio=169]{beamer}

\RequirePackage{scrbase}
\RequirePackage{graphicx}
\RequirePackage{xcolor}
\RequirePackage{tikz}
\usetikzlibrary{calc}
\usetikzlibrary{backgrounds}
\RequirePackage[normalem]{ulem}
\RequirePackage{scrbase}
\RequirePackage{tubaf-logo}
\RequirePackage{tubaf-trapezoid}

\AddToHook{begindocument/before}{
    \setbeamercolor*{palette primary}{fg=\tubaf@beamer@color@primary}
    \setbeamercolor*{palette secondary}{fg=\tubaf@beamer@color@secondary}
    \setbeamercolor*{palette tertiary}{fg=\tubaf@beamer@color@tertiary}
}

\setbeamercolor*{background}{parent=palette primary}
\setbeamercolor*{footline}{parent=palette primary}
\setbeamercolor*{structure}{parent=palette primary}
\setbeamercolor*{alerted text}{parent=palette secondary}
\setbeamercolor*{separation line}{parent=palette primary}

\setbeamercolor*{block title}{fg=white,bg=palette primary.fg,use=palette primary}
\setbeamercolor*{block body}{bg=block title.bg!15!white,use=block title}
\setbeamercolor*{block title alerted}{fg=white,bg=palette secondary.fg,use=palette secondary}
\setbeamercolor*{block body alerted}{bg=block title alerted.bg!15!white,use=block title alerted}
\setbeamercolor*{block title example}{fg=white,bg=palette tertiary.fg,use=palette tertiary}
\setbeamercolor*{block body example}{bg=block title example.bg!15!white,use=block title example}

\newcommand{\speaker}[1]{\underline{#1}}
\renewcommand{\and}{\unskip, }
\Ifnotundefined{beamer@andtitle}{\renewcommand{\beamer@andtitle}{\unskip, }}{}

\newcommand{\makeclosing}{\usebeamertemplate{closing page}}

\providecaptionname{english}{\tubaf@closing}{Thank you for your attention!}
\providecaptionname{ngerman}{\tubaf@closing}{Vielen Dank für Ihre Aufmerksamkeit!}

\newcommand{\closing}[1]{\renewcaptionname{\languagename}{\tubaf@closing}{#1}}

\providecaptionname{english}{\tubaf@contact}{%
    Technical University Bergakademie Freiberg\\
    Akademiestraße 6\\
    09599 Freiberg
}
\providecaptionname{ngerman}{\tubaf@contact}{%
    Technische Universität Bergakademie Freiberg\\
    Akademiestraße 6\\
    09599 Freiberg
}

\newcommand{\contact}[1]{\renewcaptionname{\languagename}{\tubaf@contact}{#1}}

\NewDocumentCommand\footreference{O{flush center} m}{%
    \begin{tikzpicture}[remember picture, overlay]
        \iftubaf@beamer@framenofoottext
        \node[
            anchor=south,
            align=#1,
            inner sep=\footspace,
            text width={\dimexpr\paperwidth-\Gm@lmargin-\Gm@lmargin-2\footspace\relax},
            color=fg,
            font=\usebeamerfont{footreference}
        ] at (current page.south) {#2};
        \else
        \node[
            anchor=south,
            align=#1,
            inner sep=\footspace,
            text width={\dimexpr\paperwidth-\Gm@lmargin-\Gm@lmargin\relax},
            color=fg,
            font=\usebeamerfont{footreference}
        ] at ($(current page.south) + (0,\footsize)$) {#2};
        \fi
        \usebeamercolor{footreference}
    \end{tikzpicture}
}
\let\footref\footreference

\NewDocumentCommand\sidereference{O{flush center} m}{%
    \begin{tikzpicture}[remember picture, overlay]
        \usebeamercolor{sidereference}
        \node[
            anchor=south,
            align=#1,
            rotate=90,
            inner sep=\footspace,
            text width={\dimexpr\paperheight-2\footsize\relax},
            color=fg,
            font=\usebeamerfont{sidereference}
        ] at (current page.east) {#1};
    \end{tikzpicture}
}
\let\sideref\sidereference

\newlength{\logoheadsize}
\setlength{\logoheadsize}{1.5cm}

\newlength{\logoheadmargin}
\setlength{\logoheadmargin}{0.5cm}

\setbeamercolor{title page}{parent=titlelike}
\setbeamercolor{title foot}{use=title page, fg=title page.bg, bg=title page.fg}
\if@tubaf@beamer@title@dark
\setbeamercolor{title background}{parent=title page, bg=black}
\setbeamercolor{title head}{parent=title foot}
\setbeamertemplate{title logo primary}{\resizebox{!}{\logoheadsize}{\tubaflogowhite}}
\setbeamertemplate{title logo secondary}{\insertlogo[white]{\logoheadsize}}
\else
\setbeamercolor{title background}{parent=title page, bg=white}
\setbeamercolor{title head}{parent=title page}
\setbeamertemplate{title logo primary}{\resizebox{!}{\logoheadsize}{\tubaflogo}}
\setbeamertemplate{title logo secondary}{\insertlogo{\logoheadsize}}
\fi

\ProcessOptionsBeamer

\newlength{\footsize}
\setlength{\footsize}{1cm}
\newlength{\footspace}
\setlength{\footspace}{0.15\footsize}

\newlength{\titlefootsize}
\setlength{\titlefootsize}{2.5cm}

\newlength{\closingfootsize}
\setlength{\closingfootsize}{4cm}

\setbeamertemplate{navigation symbols}{}
\setbeamertemplate{section in toc}{\usebeamercolor[fg]{local structure}\inserttocsectionnumber. \usebeamercolor[fg]{normal text}\inserttocsection}

\setbeamersize{description width=0mm}
\setlength{\leftmargini}{0.5cm}
\setlength{\leftmarginii}{0.5cm}
\setlength{\leftmarginiii}{0.5cm}

\setbeamerfont{itemize/enumerate body}{size=\normalsize}
\setbeamerfont{itemize/enumerate subbody}{size=\normalsize}
\setbeamerfont{itemize/enumerate subsubbody}{size=\normalsize}
\setbeamerfont{footline}{size=\tiny}
\setbeamertemplate{page number in head/foot}[framenumber]
\setbeamerfont{page number in head/foot}{size=\normalsize}
\setbeamerfont{closing}{parent=title}
\setbeamerfont{contact}{parent=author, size=\small}

\setbeamertemplate{itemize item}{\tikz{\usebeamercolor{local structure}\useasboundingbox (-0.05,-0.12) rectangle(0.05,0.12);\fill[color=fg] (0,0) circle(0.05)}}
\setbeamertemplate{itemize subitem}{\tikz{\usebeamercolor{local structure}\useasboundingbox (-0.05,-0.12) rectangle(0.05,0.12);\fill[color=fg] (0,0) circle(0.05)}}
\setbeamertemplate{itemize subsubitem}{\tikz{\usebeamercolor{local structure}\useasboundingbox (-0.05,-0.12) rectangle(0.05,0.12);\fill[color=fg] (0,0) circle(0.05)}}

\setbeamertemplate{bibliography item}{}
\setbeamercolor*{bibliography entry author}{}
\setbeamercolor*{bibliography entry title}{}
\setbeamercolor*{bibliography entry location}{}
\setbeamercolor*{bibliography entry note}{}

\setbeamertemplate{sidebar right}{}

\setbeamertemplate{title background}{
    \draw[fill=bg] (current page.south west) rectangle(current page.north east);
    \ifx\inserttitlegraphic\empty\else
    \node[
        anchor=center,
        text opacity=\tubaf@beamer@title@graphic@opacity,
        fill=bg
    ] at (current page.center) {\resizebox{1.01\paperwidth}{!}{\inserttitlegraphic}};
    \fi
    \tubaftrapezoidbottom[fill=fg]{\titlefootsize}
}

\setbeamertemplate{title head}{
    \ifx\inserttitle\empty\else
    \node[
        anchor=north west,
        align=flush left,
        color=fg,
        font=\usebeamerfont{title},
        name=title title,
        inner xsep=\Gm@lmargin,
        text width={\dimexpr1\paperwidth-\Gm@lmargin-\Gm@lmargin\relax}
    ] at (title head) {\inserttitle};
    \coordinate(title head) at (title title.south west);
    \fi
    \ifx\insertsubtitle\empty\else
    \node[
        anchor=north west,
        align=flush left,
        color=fg,
        font=\usebeamerfont{subtitle},
        name=title subtitle,
        inner xsep=\Gm@lmargin,
        text width={\dimexpr1\paperwidth-\Gm@lmargin-\Gm@lmargin\relax}
    ] at (title head) {\insertsubtitle};
    \coordinate(title head) at (title subtitle.south west);
    \fi
}

\setbeamertemplate{title foot}{
    \ifx\insertdate\empty\else
    \node[
        anchor=south west,
        align=flush left,
        color=fg,
        font=\usebeamerfont{date},
        name=title date,
        inner xsep=\Gm@lmargin,
        text width={\dimexpr1\paperwidth-\Gm@lmargin-\Gm@lmargin\relax}
    ] at (title foot) {\insertdate};
    \coordinate(title foot) at (title date.north west);
    \fi
    \ifx\insertinstitute\empty\else
    \node[
        anchor=south west,
        align=flush left,
        color=fg,
        font=\usebeamerfont{institute},
        name=title institute,
        inner xsep=\Gm@lmargin,
        text width={\dimexpr1\paperwidth-\Gm@lmargin-\Gm@lmargin\relax}
    ] at (title foot) {\insertinstitute};
    \coordinate(title foot) at (title institute.north west);
    \fi
    \ifx\insertauthor\empty\else
    \node[
        anchor=south west,
        align=flush left,
        color=fg,
        font=\usebeamerfont{author},
        name=title author,
        inner xsep=\Gm@lmargin,
        text width={\dimexpr1\paperwidth-\Gm@lmargin-\Gm@lmargin\relax}
    ] at (title foot) {\insertauthor};
    \coordinate(title foot) at (title author.north west);
    \fi
}

\setbeamerfont{title}{size=\LARGE}
\setbeamerfont{institute}{size=\normalsize}

\setbeamertemplate{title page}{
    \begin{tikzpicture}[remember picture, overlay]
        \useasboundingbox (current page.south west) rectangle(current page.north east);

        \node[
            anchor=north west,
            inner sep=\logoheadmargin,
            text width={\dimexpr\paperwidth-2\logoheadmargin\relax},
            align=center,
            name=title logos,
            execute at begin node = \setlength{\lineskip}{0.2\logoheadsize}
        ] at (current page.north west) {\usebeamertemplate{title logo primary} \hfill \usebeamertemplate{title logo secondary}};

        \coordinate(title head) at (title logos.south west);
        \usebeamertemplate*{title head}
        \coordinate(title foot) at ($(current page.south west) + (0, 1ex)$);
        \usebeamertemplate*{title foot}

        \begin{scope}[on background layer]
            \usebeamertemplate*{title background}
        \end{scope}
    \end{tikzpicture}
}

\setbeamerfont{section title}{size=\Large}

\setbeamertemplate{section page}{
    \begin{tikzpicture}[remember picture, overlay]
        \usebeamercolor{section title}
        \node[
            anchor=south west,
            font=\usebeamerfont{section title},
            color=fg
        ] at ($(current page.west) + (\Gm@lmargin, 0)$) {\insertsection};

        \usebeamercolor{separation line}
        \draw[color=fg] ($(current page.west) + (\Gm@lmargin, 0)$) -- ($(current page.east) - (\Gm@rmargin, 0)$);
    \end{tikzpicture}
}

\setbeamerfont{subsection title}{size=\normalsize}
\setbeamercolor{subsection title}{fg=black}

\setbeamertemplate{subsection page}{
    \begin{tikzpicture}[remember picture, overlay]
        \usebeamercolor{section title}
        \node[
            anchor=south west,
            font=\usebeamerfont{section title},
            color=fg
        ] at ($(current page.west) + (\Gm@lmargin, 0)$) {\insertsection};

        \usebeamercolor{subsection title}
        \node[
            anchor=north west,
            font=\usebeamerfont{subsection title},
            color=fg
        ] at ($(current page.west) + (\Gm@lmargin, 0)$) {\insertsubsection};

        \usebeamercolor{separation line}
        \draw[color=fg] ($(current page.west) + (\Gm@lmargin, 0)$) -- ($(current page.east) - (\Gm@rmargin, 0)$);
    \end{tikzpicture}
}

\setbeamertemplate{foottext}{\beamer@shortauthor \\[0.2em] \beamer@shorttitle} % do not use \insert... commands here as their result is not breakable

\setbeamerfont{footreference}{size=\tiny}
\setbeamercolor{footreference}{fg=darkgray}
\setbeamerfont{sidereference}{size=\tiny}
\setbeamercolor{sidereference}{fg=darkgray}

\newif\iftubaf@beamer@framenofoottext\tubaf@beamer@framenofoottextfalse
\define@key{beamerframe}{nofoottext}[true]{
    \tubaf@beamer@framenofoottexttrue
}

\apptocmd{\beamer@reseteecodes}{%
    \tubaf@beamer@framenofoottextfalse
}{}{}

\setbeamertemplate{footline}{
    \let\tubaf@beamer@oldspeaker\speaker
    \let\speaker\relax
    \vspace{\footsize}
    \begin{tikzpicture}[remember picture, overlay]
        \useasboundingbox (current page.south west) rectangle ($(current page.south east) + (0,\footsize)$);
        \node[anchor=north east, inner sep=\footspace, font=\usebeamerfont{page number in head/foot}] at ($(current page.south west) + (\Gm@lmargin,\footsize)$) {\usebeamertemplate{page number in head/foot}};
        \coordinate (footvbar south) at (\Gm@lmargin, 0);
        \coordinate (footvbar north) at (\Gm@lmargin, \footsize);
        \draw[color=fg] (footvbar south) -- (footvbar north);

        \node[anchor=south east, inner sep=\footspace, name=footseal] at (current page.south east) {\resizebox{!}{\dimexpr\footsize-\footspace\relax}{\tubafsignet}};
        \iftubaf@beamer@framenofoottext\else
        \node[anchor=south east, inner ysep=\footspace, name=footlogo] at (footseal.south west) {\insertlogo[default][\hspace{\footspace}]{\dimexpr\footsize-\footspace\relax}};
        \draw let \p{foottext west} = (footvbar south), \p{foottext east} = (footlogo.south west), \n{text width} = {\x{foottext east} - \x{foottext west} - 2\footspace} in node [
            anchor=north west,
            align=flush left,
            font=\usebeamerfont{footline},
            text width = {\dimexpr\x{foottext east} - \x{foottext west} - 0.4\footsize \relax},
            inner sep=\footspace
        ] at (footvbar north) {\usebeamertemplate{foottext}};
        \fi
    \end{tikzpicture}
    \let\spreaker\tubaf@beamer@oldspeaker
}

\setbeamertemplate{background}{
    \begin{tikzpicture}[remember picture, overlay]
        \usebeamercolor{background}
        \if@tubaf@beamer@foot@trapezoid
        \tubaftrapezoidbottom[fill=fg,opacity=\tubaf@beamer@foot@trapezoid@opacity]{\dimexpr\footsize+2mm\relax};
        \fi
    \end{tikzpicture}
}

\setbeamertemplate{closing background}{
    \draw[fill=bg] (current page.south west) rectangle(current page.north east);
    \ifx\inserttitlegraphic\empty\else
    \node[
        anchor=center,
        text opacity=\tubaf@beamer@title@graphic@opacity,
        fill=bg
    ] at (current page.center) {\resizebox{1.01\paperwidth}{!}{\inserttitlegraphic}};
    \fi
    \tubaftrapezoidbottom[fill=fg]{\closingfootsize}
}
\setbeamercolor{closing background}{parent=title background}

\setbeamertemplate{closing head}{
    \node[
        anchor=north west,
        align=left,
        color=fg,
        name=closing,
        inner xsep=\Gm@lmargin,
        text width={\dimexpr1\paperwidth-\Gm@lmargin-\Gm@lmargin\relax},
        font=\usebeamerfont{closing}
    ] at (closing head) {\tubaf@closing};
}
\setbeamercolor{closing head}{parent=title head}

\setbeamertemplate{closing foot}{
    \node[
        anchor=south west,
        align=left,
        color=fg,
        inner xsep=\Gm@lmargin,
        inner ysep=0.5cm,
        text width={\dimexpr1\paperwidth-\Gm@lmargin-\Gm@lmargin\relax},
        font=\usebeamerfont{contact}
    ] at (closing foot) {\tubaf@contact};
}
\setbeamercolor{closing foot}{parent=title foot}

\setbeamertemplate{closing page}{
    \begin{tikzpicture}[remember picture, overlay]
        \useasboundingbox (current page.south west) rectangle(current page.north east);

        \node[
            anchor=north west,
            inner sep=\logoheadmargin,
            text width={\dimexpr\paperwidth-2\logoheadmargin\relax},
            align=center,
            name=closing logos,
            execute at begin node = \setlength{\lineskip}{0.2\logoheadsize}
        ] at (current page.north west) {\usebeamertemplate{title logo primary} \hfill \usebeamertemplate{title logo secondary}};

        \coordinate(closing head) at (closing logos.south west);
        \usebeamertemplate*{closing head}
        \coordinate(closing foot) at ($(current page.south west) + (0, 1ex)$);
        \usebeamertemplate*{closing foot}

        \begin{scope}[on background layer]
            \usebeamertemplate*{closing background}
        \end{scope}
    \end{tikzpicture}
}
