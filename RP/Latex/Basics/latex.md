# LaTex

## Struktura dokumenta

LaTeXov dokument se v grobem deli na dva dela. V prvi del - `preambulo` - ponavadi vpišemo razne nastavitve našega dokumenta in uvozimo dodatne knjižnice, v drugem delu - `dokumentu` pa naše besedilo oblikujemo.

```tex
<preambula>

<dokument>
```

## Vrste LaTeX dokumentov

Določimo z `\documentclass[options]{type}` na vrhu dokumenta.

- **article:** - članek
- **amsart:** - članki za združenje American Mathematical Society
- **beamer:** - prosojnice za predavanje
- **book:** - knjiga
- **letter:** - pismo

> nekaj opcij: `a4paper,12pt`

## Oblika LaTeX dokumenta

Obliko dokumenta spreminjaš z ukazom `\pagestyle{style}`.

- **empty:** Both header and footer are empty.
- **plain:** The header is empty; the footer contains the page number.
- **headings:** The footer is empty; the header contains information determined by the class (based on sectional units) and the page number.
- **myheadings:** It is similar to headings but the user controls the information in the header.

## Knjižnice

Knjižnice uvozimo v LaTex s pomočjo `\usepackage[options]{package_name}` ukaza.

#### Splošne knjižnice:

```tex
\usepackage[slovene]{babel} % za nastavitve jezika
\usepackage[T1]{fontenc} % za nastavitve output formata
\usepackage[utf8]{inputenc} % za nastavitve kodiranja
\usepackage{hyperref}  % hiper-povezave znotraj dokumenta
```

#### Knjižnice pisav:

```tex
\usepackage{lmodern} % Boljši fonti, s pravimi šumniki
\usepackage{times} % Times New Roman
\usepackage{palatino} % Palatino
\usepackage{concrete} % uporabil Knuth v "Concrete mathematics"
```

#### Knjižnice za matematiko:

```tex
\usepackage{amsmath} % za urejanje matematičnih formul
\usepackage{amssymb} % za matematične simbole
\usepackage{amsthm}
```

**Linki:**

- [amsmath](https://www.overleaf.com/learn/latex/Aligning_equations_with_amsmath)

#### Knjižnice za risanje:

````tex
\usepackage{graphicx}
```


#### Extra knjižnice:

```tex
\usepackage{booktabs}
\usepackage{url}
````

## Macroji

```tex
\newcommand{\ime}{\content}
\newcommand{\ime}[število parametrov]{ #1 #2 ... #n parameter}
\newcommand{\ime}[2][prvi default parameter]{ #1 #2 }
```

> Pazi, da optional argumente pokličeš z `[]` ne s `{}`. S `{}` nastaviš samo obvezene parametre.

**Primeri:**

```tex
\newcommand{\numC}{\mathbb{C}}
\newcommand{\numR}{\mathbb{R}}
\newcommand{\numZ}{\mathbb{Z}}
\newcommand{\numN}{\mathbb{N}}
```

## Environmenti

> Okolja zgledajo tako, da se nastavi nekaj pred tisto kar boš noter napisal in nekaj za.

#### Narediš jih z:

```tex
\newenvironment{imeOkolja}{
  \stvariKiPridejoPredVnešeno
}{
  \stvariKiPridejoZaVnešeno
}
```

```tex
\newenvironment{imeOkolja}[število parametrov]{
  \zacetek
  #1 #2 #n
}{
  \konec
}
```

#### Uporabiš jih z:

```tex
\begin{okolje}

vse vmes

\end{okolje}
```

```tex
\begin{okolje}{prvi argument}{drugi argument}

vse vmes

\end{okolje}
```

## Definicije, Izreki

```tex
\theoremstyle{style}
\newtheorem{ime}{Printed output}[številčenje]

\newtheorem*{mydef}{Definition} % neoštevilčeno
```

**styles:**

- **definition:** boldface title, romand body. Commonly used in definitions, conditions, problems and examples.
- **plain:** boldface title, italicized body. Commonly used in theorems, lemmas, corollaries, propositions and conjectures.
- **remark:** italicized title, romman body. Commonly used in remarks, notes, annotations, claims, cases, acknowledgments and conclusions.

**linki:**

- https://www.overleaf.com/learn/latex/Theorems_and_proofs
