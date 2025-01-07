# transformer-boilerplate
Boilerplate code for a variety of transformer network architectures.
I found myself writing and rewriting a bunch of the same functions
for ML projects that use transformer networks. So I decided to just
turn it into a package that I (and anyone else) can just clone from. 

Types of transformers in this package:
1. Encoder-decoder (for translation-like tasks, i.e. go from a sequence with one vocab to a sequence drawn from a new vocab)
2. Decoder-only (for next-token prediction tasks, i.e. generate the next tokens given some input sequence)
3. Encoder-only (for sentiment analysis, fill-in-blank tasks, i.e. given the input sequence, what is the summary token)

TODO: 
The goal is to also eventually include some simple examples that
are the so-called "killer-apps" for each particular architecture. 
