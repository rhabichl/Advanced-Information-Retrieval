📘 Prompt for Extracting Legal References from Austrian Constitutional Court HTML Documents
===========================================================================================

🎯 TASK
-------

You receive a **single HTML document** from a decision of the Austrian Constitutional Court (Verfassungsgerichtshof). Your task is to **identify and extract every legal reference** contained in the **visible textual content**.

A “legal reference” means **any explicit textual reference to a legal norm or official legal document**, including:

### 1\. Statutory References (Gesetzesstellen)

*   §124 GaswirtschaftsG 2011
*   §69 GaswirtschaftsG 2011
*   §7 Abs1 E-ControlG
*   §26 Abs4 StudFG
*   §50 ElWOG

Here are all the References that are important (use only these any other can be omitted)
Aktiengesetz (AktG)
Allgemeines bürgerliches Gesetzbuch (ABGB)
Allgemeines Grundbuchsgesetz 1955 (GBG 1955)
Allgemeines Sozialversicherungsgesetz (ASVG)
Allgemeines Verwaltungsverfahrensgesetz 1991 (AVG)
Angestelltengesetz (AngG)
Arbeitsverfassungsgesetz (ArbVG)
Arbeitszeitgesetz (AZG)
Bauern-Sozialversicherungsgesetz (BSVG)
Bundesbehindertengesetz (BBG)
Bundes-Verfassungsgesetz (B-VG)
Datenschutzgesetz (DSG)
E-Government-Gesetz (E-GovG)
Einkommensteuergesetz 1988 (EStG 1988)
Epidemiegesetz 1950 (EpiG)
Europäische Menschenrechtskonvention
Führerscheingesetz (FSG)
Gewerbeordnung 1994 (GewO 1994)
Gewerbliches Sozialversicherungsgesetz (GSVG)
GmbH-Gesetz (GmbHG)
Konsumentenschutzgesetz (KSchG)
Kraftfahrgesetz 1967 (KFG 1967)
Meldegesetz 1991 (MeldeG)
Mietrechtsgesetz (MRG)
Nationalrats-Wahlordnung 1992 (NRWO)
Schulunterrichtsgesetz (SchUG)
Mai 2024 Seite 2 von 2
Sicherheitspolizeigesetz (SPG)
Staatsbürgerschaftsgesetz 1985 (StbG)
Strafgesetzbuch (StGB)
Strafprozeßordnung 1975 (StPO)
Straßenverkehrsordnung 1960 (StVO 1960)
Tierschutzgesetz (TSchG)
Umsatzsteuergesetz 1994 (UStG 1994)
Universitätsgesetz 2002 (UG)
Unternehmensgesetzbuch (UGB)
Vereinsgesetz 2002 (VerG)
Verwaltungsstrafgesetz 1991 (VStG)
Wohnungseigentumsgesetz 2002 (WEG 2002)
Zivilprozessordnung (ZPO)
Zustellgesetz (ZustG)


Include abbreviations, special characters, iVm, idF, exact punctuation, spacing, and formatting.

📌 IMPORTANT RULES
------------------

### R1 — Use only the visible text

If HTML includes both:

<span aria-hidden="true">…</span>
<span class="sr-only">…</span>

**Use only the aria-hidden text**, because this is what appears to the reader.

### R2 — Maintain perfect fidelity

Extract references **character-for-character exactly as they appear**. No:

*   rewriting
*   normalization
*   expansion of abbreviations
*   spacing changes
*   punctuation changes
*   spelling corrections

### R3 — No guessing or inferring

Only extract references that **explicitly appear** in the visible text.

### R4 — Exclude metadata headers

Do **not** treat the following as references unless they contain an actual legal reference:

*   Gericht
*   Entscheidungsdatum
*   Geschäftszahl
*   Sammlungsnummer
*   Rechtssatz / Leitsatz headings

🧠 EXTRACTION STEPS
-------------------

1.  Strip all HTML tags.
2.  Where both `aria-hidden` and `sr-only` exist, choose **aria-hidden**.
3.  Build one continuous plain-text string representing exactly what a human sees.
4.  Identify every legal reference using pattern recognition (laws, §§, BGBl, court decisions, GZ numbers, etc.).
5.  For each reference:
    *   Extract the substring **exactly as-is**
6.  Output the results **strictly in JSON**.

📦 STRICT OUTPUT FORMAT
-----------------------

Output only this JSON structure:

{
  "references": \[
    {
      "text": "<EXACT STRING AS IN HTML>",
    }
  \]
}

If no references are found:

{
  "references": \[\]
}

🚫 DO NOT
---------

*   Do not rephrase anything
*   Do not interpret meaning
*   Do not add your own notes
*   Do not output HTML
*   Do not output explanations
*   Do not output anything outside the JSON

✅ EXPECTED BEHAVIOR
-------------------

You must output:

*   **all legal references**
*   **exactly as written**
*   **character-perfect fidelity**
*   **strict JSON**
*   based on the **visible text only**
*   with correct **character indices**
