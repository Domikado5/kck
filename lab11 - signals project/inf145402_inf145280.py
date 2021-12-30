import numpy as np
import scipy.io.wavfile
import sys


def predict(filename, male_freqs=[80, 160], female_freqs=[180, 280], iterations=6):
    w, signal = scipy.io.wavfile.read(filename)

    sample_length = len(signal)/w  # length in seconds

    parts = [signal[i*w:(i+1)*w] for i in range(int(sample_length))]  # each part lasts exactly one second

    results = []

    for part in parts:
        hamming_window = np.hamming(len(part))
        data = part*hamming_window
        abs_fft = np.abs(np.fft.fft(data))/w
        fft_r = np.copy(abs_fft)
        for i in range(2, iterations):
            tab = abs_fft[::i]
            fft_r = fft_r[:len(tab)]
            fft_r *= tab
        results.append(fft_r)

    result = [0]*len(results[int(len(results)/2)])

    for res in results:
        result += res
    if sum(result[male_freqs[0]:male_freqs[1]]) > sum(result[female_freqs[0]:female_freqs[1]]):
        return 'M'
    return 'K'


if __name__ == '__main__':
    args = sys.argv
    filename = args[1]

    prediction = predict(filename)

    print(prediction)