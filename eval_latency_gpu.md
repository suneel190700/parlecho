# Parlecho streaming latency — whisper-small + CT2 int8 NLLB + XTTS-v2, greedy decoding

Input: clip.mp3, es->en, device=cuda, real-time paced. Lag = stream-time end of speech to dubbed chunk ready; includes ~0.42s VAD end-of-speech detection. Models resident before stream start (load: 51.9s, excluded).

**n=14 utterances — p50 lag 0.83s, p95 lag 1.60s**

| End (s) | Audio (s) | Lag (s) | Queue (s) | ASR | Translate | TTS | Output |
|---------|-----------|---------|-----------|-----|-----------|-----|--------|
| 3.17 | 2.11 | 1.6 | 0.07 | 0.18 | 0.07 | 0.86 | Hey, good morning, you. |
| 6.01 | 1.86 | 0.75 | 0.03 | 0.05 | 0.02 | 0.23 | What's your name? |
| 8.57 | 1.5 | 0.74 | 0.03 | 0.03 | 0.01 | 0.25 | Paul or Paul? |
| 10.59 | 1.66 | 0.85 | 0.02 | 0.03 | 0.01 | 0.37 | Paula, with you. |
| 12.89 | 0.86 | 0.93 | 0.02 | 0.03 | 0.02 | 0.44 | What about your last name? |
| 14.53 | 1.12 | 0.7 | 0.02 | 0.02 | 0.02 | 0.22 | - I'm not a soda. |
| 18.17 | 2.3 | 0.78 | 0.04 | 0.04 | 0.01 | 0.27 | Where are you from? |
| 20.32 | 1.57 | 0.83 | 0.0 | 0.04 | 0.02 | 0.35 | Oh, Brazilian, that's good! |
| 27.84 | 6.3 | 1.47 | 0.0 | 0.09 | 0.04 | 0.89 | And how old are you? 19 19 What are you doing? I'm |
| 29.73 | 0.9 | 0.71 | 0.0 | 0.03 | 0.02 | 0.25 | And you have a cell phone? |
| 36.64 | 6.59 | 1.52 | 0.0 | 0.08 | 0.02 | 1.0 | Yes, my number is 675312908. |
| 44.99 | 6.5 | 0.83 | 0.0 | 0.07 | 0.03 | 0.32 | I'm not sure what the hell happened to you. |
| 49.73 | 1.02 | 0.75 | 0.0 | 0.03 | 0.02 | 0.29 | All right, thank you. |
| 53.02 | 2.91 | 1.25 | 0.0 | 0.04 | 0.04 | 0.75 | Well, look, sit here for a moment and you're getti |

Streaming v1 scope: no source separation, single rolling clone reference, greedy decoding.
