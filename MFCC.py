"""
Acoustic analysis
Mel-Frequency Cepstral Coefficients algorithm implementation
Created for lectures of KKY/ARŘ at University of West Bohemia
"""

import numpy as np
from scipy.io import wavfile
import scipy.fftpack as fft
import scipy.signal

"""
Audio normalization implemented to mimic Matlab normalization
"""
def normalize_audio(original_audio):
    #normalized_audio = original_audio / np.max(np.abs(original_audio))
    normalized_audio = original_audio / 2**15
    return normalized_audio

"""
Convert frequency in Hertz to frequency in Mel
"""
def hz2mel(frequency):
    mel_frequency = 2595 * np.log10(1+frequency/700)
    return mel_frequency


"""
Convert frequency in Mel to frequency in Hertz
"""
def mel2hz(mel_frequency):
    frequency = 700*((10**(mel_frequency/2595))-1)
    return frequency


"""
Table of all relevant points of all the triangular filters
Each filter starts at point (filter_id -1), peaks at point (filter_id) and ends at point (filter_id+1)
"""
def filter_table(max_frequency_Hz = 4000, number_of_filters = 15):
    step = hz2mel(max_frequency_Hz)/(number_of_filters+1)
    filter_centers_mel = np.zeros(number_of_filters+2)
    filter_centers_Hz = np.zeros(number_of_filters+2)
    center = step
    for i in range(1,number_of_filters+2):
        filter_centers_mel[i] = center
        filter_centers_Hz[i] = mel2hz(center)
        center = center + step
        
    """
    for i,h,m in zip(range(1,17),filter_centers_Hz,filter_centers_mel):
        print("{i} \t {m:.4f} \t {h:.4f}".format(i=i,h=h,m=m))
    """
    return filter_centers_Hz


"""
Formula to calculate the value of the chosen filter at the chosen frequency
"""
def filter_function(frequency,filter_start,filter_center,filter_end):
    if ((filter_start<=frequency) and (frequency<filter_center)):
        return ((frequency-filter_start)/(filter_center-filter_start))
    elif ((filter_center<=frequency) and (frequency<filter_end)):
        return ((frequency-filter_end)/(filter_center-filter_end))
    else: 
        return 0


"""
Value of chosen triangular filter at a given frequency calculated with a corresponding formula
"""
def filter_value(frequency,filter_table,filter_id=1):
    filter_start = filter_table[filter_id-1]
    filter_center = filter_table[filter_id]
    filter_end = filter_table[filter_id+1]
    
    filter_value = filter_function(frequency,filter_start,filter_center,filter_end)
    return filter_value

"""
Split source audio into segments of size 'segment_size' with a shift between segments the size of 'shift'
"""
def microsegments(sampling_frequency, audio_signal, segment_size = 256, shift = 10):
    shift_size = shift / 1000 * sampling_frequency
    number_of_segments = int((len(audio_signal) - segment_size) / shift_size) + 1
    
    segments = []
    start = 0
    end = start + segment_size
    for i in range(number_of_segments):   
        segments.append(audio_signal[start:end])
        start = start + int(shift_size) 
        end = start + segment_size
    return segments

"""
Bulk method for extracting Mel-frequency cepstral coefficients froma audiosignal
"""
def mfcc(audio_path, segment_size = 256, shift = 10, max_frequency_Hz = 4000, number_of_filters = 15):
    sampling_frequency, audio = wavfile.read(audio_path)
    normalized_audio = normalize_audio(audio)
    segments = microsegments(sampling_frequency, normalized_audio)
    
    half_segment_size = int(segment_size/2)
    all_filter_values = np.zeros((number_of_filters,half_segment_size))
    step = int(max_frequency_Hz/half_segment_size)
    table = filter_table(max_frequency_Hz,number_of_filters)
    for i in range(number_of_filters):
        f = 0
        value = filter_value(f,table,i+1)
        for j in range(half_segment_size):
            all_filter_values[i,j] = value
            f += step
            value = filter_value(f,table,i+1)

    mfcc_table = []
    window = scipy.signal.get_window("hamm", segment_size, fftbins=True)
    for segment in segments:
        windowed_segment = segment * window 
        segment_fft = fft.fft(windowed_segment)
        absolute_value = np.abs(segment_fft)
        first_half = absolute_value[0:half_segment_size] #we only need the first half as the fft is symmetric by y-axis
        product = np.dot(all_filter_values,first_half)
        log_result = np.log10(product)
        cos_result = fft.dct(log_result,norm='ortho') #ortho is used to mimic matlab dct functionality

        mfcc_table.append(cos_result)
    return mfcc_table


if __name__ == "__main__":
    
    AUDIO_PATH = "mfcc/00010001.wav"
    WANT_CSV_EXPORT = False

    mel_frequency_cepstral_coefficients = mfcc(AUDIO_PATH)
    
    if WANT_CSV_EXPORT:
        #export to csv
        import csv
        with open("./out.csv", 'w') as csvfile: 
            csvwriter = csv.writer(csvfile) 
            for row in mel_frequency_cepstral_coefficients:
                csvwriter.writerow(row)