# Parlecho streaming latency — whisper-small + CT2 int8 NLLB + XTTS-v2, greedy decoding

Input: Personal Information - Example 1.mp3, es->en, real-time paced. Lag = stream-time end of speech to dubbed chunk ready; includes ~0.42s VAD end-of-speech detection. Models resident before stream start (load: 13.7s, excluded).

**n=14 utterances — p50 lag 3.49s, p95 lag 6.36s**

| End (s) | Audio (s) | Lag (s) | Queue (s) | ASR | Translate | TTS | Output |
|---------|-----------|---------|-----------|-----|-----------|-----|--------|
| 3.17 | 2.11 | 3.49 | 0.0 | 0.89 | 0.2 | 1.96 | Hey, good morning, you. |
| 6.01 | 1.86 | 2.58 | 0.22 | 0.92 | 0.14 | 0.87 | What's your name? |
| 8.57 | 1.5 | 3.07 | 0.0 | 0.83 | 0.09 | 1.72 | Paolo or Paulo? |
| 10.59 | 1.66 | 3.13 | 0.63 | 0.79 | 0.11 | 1.17 | Paula, with you. |
| 12.89 | 0.86 | 2.98 | 0.4 | 0.81 | 0.12 | 1.23 | What about your last name? |
| 14.53 | 1.12 | 3.76 | 0.93 | 0.79 | 0.15 | 1.47 | - I'm not a soda. |
| 18.17 | 2.3 | 2.33 | 0.01 | 0.85 | 0.11 | 0.94 | Where are you from? |
| 20.32 | 1.57 | 2.96 | 0.0 | 0.86 | 0.18 | 1.49 | Oh, Brazilian, that's good! |
| 27.84 | 6.3 | 6.2 | 0.0 | 1.0 | 0.22 | 4.52 | And how old are you? 19. 19. What are you doing? |
| 29.73 | 0.9 | 6.36 | 3.89 | 0.81 | 0.13 | 1.11 | And you have a cell phone? |
| 36.64 | 6.59 | 6.12 | 0.0 | 0.87 | 0.19 | 4.63 | Yes, my number is 675312908. |
| 44.99 | 6.5 | 3.91 | -0.0 | 0.99 | 0.27 | 2.24 | I'm going to have to go to the hospital and get so |
| 49.73 | 1.02 | 2.61 | 0.0 | 0.82 | 0.12 | 1.25 | All right, thank you. |
| 53.02 | 2.91 | 4.61 | -0.0 | 0.89 | 0.29 | 3.01 | Well, look, sit here for a moment and you're being |

Hardware: Apple M-series CPU. Streaming v1 scope: no source separation, single rolling clone reference, greedy decoding.
