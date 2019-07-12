from vdataset import *


def wav_encode(name):
    
    f0s = list() 
    res = glob.glob('/data/tree/qinghua/train/*%s*.wav' % name)
    num = 0
    for f in res:
        x, fs = librosa.load(f, dtype=np.float64, sr=16000)
        _f0, t = pw.dio(x, fs)    
        f0 = pw.stonemask(x, _f0, t, fs)  
        f0s.append(f0)
        num += 1
        if num % 100 == 0:
            print(num)

    means, stds = logf0_statistics(f0s)

    print(means, stds)
    return means, stds
    
    
def main():
    m1, s1 = wav_encode('4_')
    m2, s2 = wav_encode('8_')
    statistics = np.array([[m1,s1], [m2, s2]])
    np.save('./data/statistics.npy', statistics)
    
    

if __name__ == '__main__':
    main()
