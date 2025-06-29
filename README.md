# TPTSI

## Dataset

### Detailed statistics of benchmark dataset

| Dataset         | #Tar    | %M   | #Sent  | #Len    |
|-----------------|---------|------|--------|---------|
| MOH(train) | 1,489   |22.49 | 1,482  | 7.31 |
& test  & 150     & 50.00 & 150    & 7.26 |
& val  & 210     & 50.00    & 70    & 7.17 |
& train & 1772    & 52.65 & 1,697  & 28.06 |
& test  & 1965    & 61.67 & 1,925  & 29.05 |
& val  & 650     & 60.92    & 647    & 28.88 |
& train & 160,154 & 11.97 & 10,909 & 27.59 |
& test  & 22,196  & 17.94 & 3,601  & 28.1|
& val  & 6658     & 17.99    & 1996    & 28.39 |


We use three well-known public English datasets. The VU Amsterdam Metaphor Corpus (VUA) has been released in metaphor detection shared tasks in 2018 and 2020. We use one version of VUA datasets, called <b>VUA-20</b>. We split VUA-18 and VUA-20 each for training, validation, and test datasets. VUA-20 includes VUA-18, and VUA-Verb (test) is a subset of VUA-18 (test) and VUA-20 (test). We also use VUA datasets categorized into different POS tags (verb, noun, adjective, and adverb) and genres (news, academic, fiction, and conversation).<br>
We employ <b>MOH-X</b> and <b>TroFi</b> for testing only. 
