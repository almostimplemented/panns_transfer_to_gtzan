sample_rate = 16000
clip_samples = sample_rate * 15

mel_bins = 64
fmin = 50
fmax = 8000
window_size = 512
hop_size = 160
window = 'hann'
pad_mode = 'reflect'
center = True
device = 'cuda'
ref = 1.0
amin = 1e-10
top_db = None

classes_num = 120
