import glob
import librosa
import numpy as np
import pyworld as pw
import torch
import h5py

dim = 24
SR = 16000

def savefile(path, csp):
    f = h5py.File(path,'w')
    f['mcep'] = csp
    f.close()

def getfile(path):
    f = h5py.File(path,'r')
    csp = f['mcep'][:]
    f.close()
    return csp

def getF0(filename):
    x, fs = librosa.load(filename, dtype=np.float64, sr=SR)
    f0,sp,ap= pw.wav2world(x,fs)
    E = np.sum(sp,axis=1)

    f0 = np.maximum(f0, 1)
    E  = np.maximum(E,1e-5)
    
     
    return np.log(f0), np.log(E)

def getdata_voice():
    ID = torch.load('/home/tree/data/LJSpeech/traindata/ID.pt')
    path = '/home/tree/data/LJSpeech/wavs/%s.wav'

    dataset = np.zeros((1,26))

    num = 0

    for x in ID:
        filename = path % x
        f0, E = getF0(filename)
        
        f0 = f0.reshape((f0.shape[0], 1))
        E = E.reshape((E.shape[0], 1))
        imgname = filename.replace('wavs','imgs').replace('wav','h5')
        csp = getfile(imgname)

        res= np.concatenate((f0,E,csp),axis=1)
        dataset = np.concatenate((dataset,res),axis=0)
        num += 1
        print(num, dataset.shape)
    torch.save(dataset, '/nas/jy/LJSpeech/vdata.pt')

def getdata_csp():
    ID = torch.load('/home/tree/data/LJSpeech/traindata/ID.pt')
    path = '/home/tree/data/LJSpeech/imgs/%s.h5'

    dataset = np.zeros((1,24))

    num = 0

    for x in ID:
        filename = path % x
        
        csp = getfile(filename)

        dataset = np.concatenate((dataset,csp),axis=0)
        num += 1
        print(num, dataset.shape)
    torch.save(dataset, '/nas/jy/LJSpeech/csp_data.pt')

def mcep(filename):
    x, fs = librosa.load(filename, dtype=np.float64, sr=SR)
    f0,sp,ap= pw.wav2world(x,fs)
    csp = pw.code_spectral_envelope(sp,fs,dim)
    print(csp.shape)
    return csp

def getdata():
    ID = torch.load('traindata/ID.pt')
    path = 'wavs/%s.wav'
    for x in ID:
        filename = path % x
        res = mcep(filename)
        imgname = filename.replace('wavs','imgs').replace('wav','h5')
        savefile(imgname, res)
        print(imgname)
    

def get_CSP_Norm():
    res = glob.glob('imgs/*.h5')
    num = 0 
    smean = 0
    sstd = 0
    for f in res:
        d = h5py.File(f,'r')
        csp = d['mcep']
        m = np.mean(csp)
        s = np.std(csp)
        print('num:', num, m, s)
        num += 1 
        smean += m
        sstd += s

    torch.save((smean/num, sstd/num), 'traindata/statics.pt')
    

if __name__ == '__main__':
    getdata_csp()
