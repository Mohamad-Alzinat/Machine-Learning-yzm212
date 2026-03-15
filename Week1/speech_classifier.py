
import numpy as np
from hmmlearn import hmm

# "EV" kelimesi için HMM modeli
model_ev = hmm.MultinomialHMM(n_components=2)

model_ev.startprob_ = np.array([1.0, 0.0])

model_ev.transmat_ = np.array([
    [0.6, 0.4],
    [0.2, 0.8]
])

model_ev.emissionprob_ = np.array([
    [0.7, 0.3],  # e durumu
    [0.1, 0.9]   # v durumu
])

# "OKUL" kelimesi için örnek HMM modeli
model_okul = hmm.MultinomialHMM(n_components=2)

model_okul.startprob_ = np.array([0.5,0.5])

model_okul.transmat_ = np.array([
    [0.7,0.3],
    [0.3,0.7]
])

model_okul.emissionprob_ = np.array([
    [0.6,0.4],
    [0.4,0.6]
])

def kelime_tahmin(obs):

    obs = np.array(obs).reshape(-1,1)

    score_ev = model_ev.score(obs)
    score_okul = model_okul.score(obs)

    if score_ev > score_okul:
        return "EV"
    else:
        return "OKUL"

# Örnek test
observation = [0,1]

print("Tahmin:", kelime_tahmin(observation))
