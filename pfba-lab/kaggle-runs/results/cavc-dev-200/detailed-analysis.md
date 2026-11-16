# CAVC 200-sentence dev: detailed per-sentence analysis

Run: `--mode dev --limit 200` on Kaggle (kernel `pfba-cavc-v3-dev`), holdout = dev hash bucket 1 among the first 200 sentences (n=17). Controller and fixed `(4,6)` share runtime, translator, RALCP and timing. Sentence BLEU is sacrebleu with `effective_order=True`; corpus BLEU is the official default signature.

## Aggregate

| metric | controller | fixed (4,6) | delta |
|---|---:|---:|---:|
| corpus BLEU | 27.72 | 20.40 | +7.33 |
| LAAL mean / median | 3.41 / 2.96 | 2.38 / 1.90 | +1.03 mean |
| completion_s mean / median | 9.98 / 8.05 | 9.40 / 8.44 | +0.58 mean / -1.00 median |
| READ fraction | 0.397 | 0 | - |

Wins (controller better): LAAL 2/17, completion 11/17, sentence BLEU 13/17, identical hypothesis 2/17.

`paired.json` bootstrap (n=17): BLEU +9.12 [2.37, 16.47]; LAAL +1.03 [0.53, 1.58] (controller worse); completion_s +0.58 [-2.10, 3.40] (not significant).

## Matched-LAAL subsets

| subset | n | corpus BLEU ctl | fixed | delta | LAAL ctl / fix | time ctl / fix | LAAL win count |
|---|---:|---:|---:|---:|---|---|---:|
| all | 17 | 27.72 | 20.40 | +7.33 | 3.41 / 2.38 | 9.98 / 9.40 | 2/17 |
| abs(dLAAL) <= 0.25 | 5 | 40.94 | 36.44 | +4.50 | 1.99 / 1.99 | 13.68 / 8.92 | 2/5 |
| abs(dLAAL) <= 0.50 | 7 | 39.36 | 32.28 | +7.08 | 2.18 / 2.07 | 11.52 / 8.42 | 2/7 |
| abs(dLAAL) <= 1.00 | 11 | 38.72 | 28.08 | +10.64 | 2.57 / 2.19 | 8.57 / 6.81 | 2/11 |
| ctl LAAL <= fix | 2 | 29.35 | 19.75 | +9.60 | 2.23 / 2.47 | 13.91 / 10.84 | 2/2 |
| ctl LAAL <= fix + 0.5 | 7 | 39.36 | 32.28 | +7.08 | 2.18 / 2.07 | 11.52 / 8.42 | 2/7 |
| completion win | 11 | 22.39 | 10.12 | +12.27 | 3.96 / 2.75 | 7.24 / 10.01 | 1/11 |
| completion win & dLAAL<=0.5 | 3 | 27.57 | 17.40 | +10.18 | 2.77 / 2.57 | 8.75 / 9.92 | 1/3 |
| LAAL win & BLEU win | 2 | 29.35 | 19.75 | +9.60 | 2.23 / 2.47 | 13.91 / 10.84 | 2/2 |

**Pareto-dominant sentences (LAAL <= fixed, completion <= fixed, different hypothesis): 1/17** (dev:0162).

## Zero-BLEU-change sentences that cost more time

| sentence | dLAAL | dTime | controller actions |
|---|---:|---:|---|
| dev:0021 | +0.15 | +9.77 | b4h6:1 b6h10:9 |
| dev:0124 | +0.23 | +1.36 | b6h10:2 read:1 |
| dev:0187 | +0.07 | +6.52 | b4h4:1 b6h10:9 read:4 |

Total extra time on zero-BLEU-change sentences: +17.65 s. This is the main driver of the positive mean completion delta.

## Per-sentence table

| sentence | src words | LAAL ctl | LAAL fix | dLAAL | time ctl | time fix | dTime | BLEU ctl | BLEU fix | dBLEU | same hyp | controller actions |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| dev:0009 | 8 | 2.70 | 1.90 | +0.80 | 5.55 | 4.58 | +0.97 | 40.90 | 8.55 | +32.35 | False | b2h4:1 b4h6:1 b6h10:2 read:1 |
| dev:0021 | 13 | 1.49 | 1.34 | +0.15 | 18.21 | 8.44 | +9.77 | 60.08 | 60.08 | +0.00 | True | b4h6:1 b6h10:9 |
| dev:0051 | 12 | 5.64 | 4.31 | +1.34 | 6.82 | 7.82 | -1.00 | 18.95 | 3.70 | +15.25 | False | b2h10:3 b6h10:2 read:4 |
| dev:0074 | 6 | 2.46 | 1.71 | +0.75 | 1.55 | 3.00 | -1.46 | 70.71 | 27.05 | +43.66 | False | b2h4:2 read:1 |
| dev:0075 | 7 | 2.83 | 2.00 | +0.83 | 2.54 | 3.71 | -1.16 | 27.78 | 23.36 | +4.42 | False | b6h10:1 read:3 |
| dev:0080 | 23 | 4.15 | 2.85 | +1.30 | 14.23 | 16.44 | -2.21 | 24.51 | 13.28 | +11.23 | False | b2h10:9 b6h10:4 read:7 |
| dev:0088 | 24 | 7.50 | 4.04 | +3.46 | 8.63 | 18.32 | -9.69 | 6.89 | 5.72 | +1.18 | False | b2h10:4 b6h10:2 read:15 |
| dev:0099 | 20 | 2.96 | 0.52 | +2.44 | 8.67 | 13.93 | -5.25 | 9.80 | 4.82 | +4.98 | False | b2h10:9 b6h10:1 read:7 |
| dev:0108 | 8 | 4.95 | 4.00 | +0.95 | 3.94 | 4.65 | -0.71 | 15.85 | 15.19 | +0.66 | False | b2h10:2 b6h10:1 read:2 |
| dev:0124 | 6 | 2.09 | 1.86 | +0.23 | 4.34 | 2.98 | +1.36 | 29.06 | 29.06 | +0.00 | True | b6h10:2 read:1 |
| dev:0125 | 22 | 4.70 | 1.45 | +3.25 | 30.14 | 15.99 | +14.14 | 6.20 | 29.30 | -23.10 | False | b2h4:1 b6h10:15 read:3 |
| dev:0128 | 10 | 3.61 | 3.14 | +0.47 | 4.19 | 5.86 | -1.66 | 13.83 | 2.84 | +10.99 | False | b2h10:4 read:3 |
| dev:0134 | 13 | 1.74 | 1.36 | +0.38 | 8.05 | 8.49 | -0.44 | 46.38 | 20.60 | +25.78 | False | b6h10:4 read:6 |
| dev:0157 | 18 | 4.72 | 3.15 | +1.57 | 6.97 | 12.43 | -5.46 | 15.81 | 10.76 | +5.05 | False | b2h10:1 b6h10:3 read:11 |
| dev:0162 | 21 | 2.97 | 3.21 | -0.24 | 14.00 | 15.42 | -1.42 | 19.65 | 18.96 | +0.69 | False | b6h10:7 read:11 |
| dev:0186 | 10 | 1.50 | 1.73 | -0.23 | 13.82 | 6.26 | +7.56 | 42.31 | 20.46 | +21.86 | False | b6h10:7 |
| dev:0187 | 17 | 1.89 | 1.83 | +0.07 | 18.03 | 11.51 | +6.52 | 47.05 | 47.05 | +0.00 | False | b4h4:1 b6h10:9 read:4 |

## Hypotheses where the two policies differ

### dev:0009 (dLAAL +0.80, dTime +0.97, dBLEU +32.35)

- SRC: Dies bestätigt auch Peter Arnold vom Landratsamt Offenburg.
- REF: This was also confirmed by Peter Arnold from the Offenburg District Office.
- CTL: This is also confirmed by Peter Arnold of the Offenburg District Council.
- FIX: This also confirms that Peter Arnold of the Landratsamt Offenburg.

### dev:0051 (dLAAL +1.34, dTime -1.00, dBLEU +15.25)

- SRC: Die Art und Weise, wie wir diese Steuern zahlen, wird sich verändern.
- REF: There is going to be a change in how we pay these taxes.
- CTL: The way we pay these taxes will change.
- FIX: The way the taxes are paid will be changed.

### dev:0074 (dLAAL +0.75, dTime -1.46, dBLEU +43.66)

- SRC: Nevada hat bereits ein Pilotprojekt abgeschlossen.
- REF: Nevada has already completed a pilot.
- CTL: Nevada has already completed a pilot project.
- FIX: Nevada has already launched a pilot project.

### dev:0075 (dLAAL +0.83, dTime -1.16, dBLEU +4.42)

- SRC: New York City erwägt ebenfalls ein solches.
- REF: New York City is looking into one.
- CTL: New York City is also considering such a thing.
- FIX: New York City Considers one as well.

### dev:0080 (dLAAL +1.30, dTime -2.21, dBLEU +11.23)

- SRC: „Bedenken gegen Big Brother und derartige Dinge waren ein großes Problem“, erklärt Alauddin Khan, Leiter des Strategie- und Ergebnismanagements im Verkehrsministerium von Nevada.
- REF: "Concerns about Big Brother and those sorts of things were a major problem," said Alauddin Khan, who directs strategic and performance management at the Nevada Department of Transportation.
- CTL: "Confronting Big Brother and such things were a big problem," explains Alauddin Khan, head of the Strategy and Results Management department at the Nevada Department of Transport.
- FIX: "Confronting Big Brother and such things were a big problem," explains Alauddin Khan, the head of the strategy and result management in the Ministry of Transport.

### dev:0088 (dLAAL +3.46, dTime -9.69, dBLEU +1.18)

- SRC: Vielmehr war es ihr Ziel, in einem aufstrebenden Markt für Kfz-Versicherungen Fuß zu fassen, bei denen Fahrer auf Grundlage der gefahrenen Meilen zahlen sollen.
- REF: It was seeking to break into an emerging market in auto insurance, in which drivers would pay based on their mileage.
- CTL: Rather, it was their aim to gain a foothold in an emerging market for car insurance, where drivers can pay on the basis of the miles they have driven .
- FIX: The aim was to create a market for motor insurance , where drivers are based on the miles they were driving.

### dev:0099 (dLAAL +2.44, dTime -5.25, dBLEU +4.98)

- SRC: Einige Verkehrsplaner fragen sich allerdings, ob das ganze Gerede über das Bezahlen pro Meile nicht nur ein riesiges Ablenkungsmanöver sei.
- REF: Some transportation planners, though, wonder if all the talk about paying by the mile is just a giant distraction.
- CTL: Some traffic planners ask , however, whether the whole talk about paying per mile is not just a huge diversion of the way.
- FIX: Some traffic planners ask themselves , however, whether the whole talk about the payment of per mile not only a huge diversion .

### dev:0108 (dLAAL +0.95, dTime -0.71, dBLEU +0.66)

- SRC: Dort soll beim Sägen ein Kurzschluss entstanden sein.
- REF: A short-circuit was reported to have occurred there during sawing.
- CTL: There is supposed to have been a short circuit during sawing.
- FIX: There, a short circuit was created during sawing.

### dev:0125 (dLAAL +3.25, dTime +14.14, dBLEU -23.10)

- SRC: Auf seinem Besuch im Landkreis Breisgau-Hochschwarzwald ließ sich Landtagspräsident Guido Wolf auch in Titisee-Neustadt darüber aufklären, woher die Arbeitskräfte von morgen kommen.
- REF: On a visit to the district of Breisgau-Hochschwarzwald, it was explained to Parliamentary President Guido Wolf in Titisee-Neustadt where the workforce of tomorrow will come from.
- CTL: On his visit, the President of the Parliament Guido Wolf also informed Titisee-Neustadt about where the workers of tomorrow came from.
- FIX: On his visit to the district of Breisgau-Hochschwarzwald , President of the Landtag Guido Wolf also informed Titisee-Neustadt about where the workers of tomorrow came from.

### dev:0128 (dLAAL +0.47, dTime -1.66, dBLEU +10.99)

- SRC: Dabei zeigte sich Wolf beeindruckt über die Modellprojekte zur Ausbildung.
- REF: During the presentation, Wolf seemed impressed by the educational pilot project.
- CTL: Wolf was impressed by the model projects for the training.
- FIX: The Wolf was impressed.

### dev:0134 (dLAAL +0.38, dTime -0.44, dBLEU +25.78)

- SRC: Aber niemand könne es sich leisten, qualifizierte Arbeitskräfte zu verlieren, sagte er weiter.
- REF: However, nobody can afford to lose qualified workers," he added.
- CTL: But no one can afford to lose qualified workers, he continued.
- FIX: But no one can afford to lose skilled labour, said he.

### dev:0157 (dLAAL +1.57, dTime -5.46, dBLEU +5.05)

- SRC: Ursprünglich war die Schulhofsanierung sogar schon in den Jahren 2008/2009 geplant, doch hohe unplanmäßige Ausgaben brachten eine Verschiebung.
- REF: The school yard renovation was originally planned back in 2008/2009, however, high unplanned expenses meant that the work had to be pushed back.
- CTL: Originally, the schoolyard renovation was planned in 2008/2009, but high unscheduled expenditures caused a shift.
- FIX: The schoolyard renovation was originally already in the years 2008/2009 planned, but high unplanned expenditures brought a postponement.

### dev:0162 (dLAAL -0.24, dTime -1.42, dBLEU +0.69)

- SRC: In die anhaltenden Diskussionen um die Zukunft der Musikhochschulen in Baden-Württemberg hat sich nun auch Cem Özdemir, Bundesvorsitzender der Grünen, eingeschaltet.
- REF: Cem Özdemir, Federal Chairman of the Green Party, has now engaged in the ongoing discussions regarding the future of the Baden-Wuerttemberg Music Academy.
- CTL: In the ongoing discussions on the future of the music colleges in Baden-Württemberg, Cem Özdemir, the chairman of the Greens, has now also taken part.
- FIX: In the ongoing discussions on the future of the music colleges in Baden-Württemberg, the Cem Özdemir, the chairman of the Greens, has now also taken part.

### dev:0186 (dLAAL -0.23, dTime +7.56, dBLEU +21.86)

- SRC: Allen dreien droht als Höchststrafe lebenslanger Freiheitsentzug, wie Beamte sagten.
- REF: All three face a maximum penalty of life imprisonment, according to the authorities.
- CTL: All three are threatened as a maximum penalty of life imprisonment, as officials said.
- FIX: All three are threatened as a maximum sentence of life imprisonment, as the officials said.

### dev:0187 (dLAAL +0.07, dTime +6.52, dBLEU +0.00)

- SRC: Seit 2006 seien acht derartige Tunnel entdeckt worden, hieß es auf der Pressekonferenz in San Diego weiter.
- REF: Since 2006, eight tunnels of this type have been discovered, the press conference in San Diego continued.
- CTL: Since 2006, eight such tunnels have been discovered , which was reported at the press conference in San Diego.
- FIX: Since 2006, eight such tunnels have been discovered , it was said on the press conference in San Diego .

