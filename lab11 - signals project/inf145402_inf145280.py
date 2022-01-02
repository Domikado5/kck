import numpy as np
import numpy.typing as npt
import scipy.io.wavfile
import sys
from typing import List


def get_frequencies(
    parts: npt.ArrayLike, sample_rate: int, iterations: int = 6
) -> np.ndarray:
    freq = []

    for part in parts:
        hamming_window = np.hamming(len(part))
        data = part * hamming_window
        abs_fft = np.abs(np.fft.fft(data)) / sample_rate
        fft_r = np.copy(abs_fft)
        for i in range(2, iterations):
            tab = abs_fft[::i]
            fft_r = fft_r[: len(tab)]
            fft_r *= tab
        freq.append(fft_r)
    return freq


def is_male(
    freq_count: npt.ArrayLike,
    male_freqs: List[int] = [80, 160],
    female_freqs: List[int] = [180, 280],
) -> bool:
    if sum(freq_count[male_freqs[0] : male_freqs[1]]) > sum(
        freq_count[female_freqs[0] : female_freqs[1]]
    ):
        return True
    return False


def predict(filename: str) -> str:
    try:
        sample_rate, signal = scipy.io.wavfile.read(filename)
    except:
        print("Błąd w odczycie pliku")
        return

    sample_length = len(signal) / sample_rate  # length in seconds

    parts = []
    for i in range(int(sample_length)):  # each part lasts exactly one second
        parts.append(signal[i * sample_rate : (i + 1) * sample_rate])

    frequencies = get_frequencies(parts, sample_rate)

    result = [0] * len(frequencies[int(len(frequencies) / 2)])

    for freq in frequencies:
        result += freq

    if is_male(result):
        return "M"
    return "K"


if __name__ == "__main__":
    args = sys.argv
    filename = args[1]

    prediction = predict(filename)

    if prediction:
        print(prediction)
