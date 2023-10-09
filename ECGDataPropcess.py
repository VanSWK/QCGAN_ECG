# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 18:42:38 2020

@author: Aiyun
"""
import numpy as np
import wfdb
import matplotlib.pyplot as plt
from scipy.signal import resample
import glob
import pywt

DATA_FREQ = 360  # 原始心率采样频率
TAR_DIM = 256  # 心波信号目标维度

S1 = int(DATA_FREQ * 0.4)  # 截取R峰的前0.4s和后0.5s为一个心波
S2 = int(DATA_FREQ * 0.5)


# --------------------小波去噪-----------------
def WTfilt_1d(sig):
    """
    对信号进行小波变换滤波
    :param sig: 输入信号，1-d array
    :return: 小波滤波后的信号，1-d array
    """
    coeffs = pywt.wavedec(sig, 'db6', level=9)
    coeffs[-1] = np.zeros(len(coeffs[-1]))
    coeffs[-2] = np.zeros(len(coeffs[-2]))
    coeffs[0] = np.zeros(len(coeffs[0]))
    sig_filt = pywt.waverec(coeffs, 'db6')
    return sig_filt


# -------------------------心拍截取-------------------
def heartbeat(file0):
    '''
    file0:下载的MITAB数据
    
    '''

    AAMI_dic = {'N': ['N', 'L', 'R'],
                'S': ['A', 'J', 'S', 'a', 'e', 'j'],
                'V': ['E', 'V'],
                'F': ['F'],
                'Q': ['P', 'Q', 'f', '/', 'f']}

    N_Seg = []
    SVEB_Seg = []
    VEB_Seg = []
    F_Seg = []
    Q_Seg = []
    # --------去掉指定的四个导联的头文件---------
    De_file = [panth[:-1] + '\\102.hea', panth[:-1] + '\\104.hea', panth[:-1] + '\\107.hea', panth[:-1] + '\\217.hea']
    file = list(set(file0).difference(set(De_file)))

    for f in range(len(file)):
        annotation = wfdb.rdann(panth + file[f][-7:-4], 'atr')
        record_name = annotation.record_name  # 读取记录名称
        Record = wfdb.rdsamp(panth + record_name)[0][:, 0]  # 一般只取一个导联
        record = WTfilt_1d(Record)  # 小波去噪
        label = annotation.symbol  # 心拍标签列表
        label_index = annotation.sample  # 标签索引列表
        for j in range(len(label_index)):
            if label_index[j] >= S1 and (label_index[j] + S2) <= 650000:
                if label[j] in AAMI_dic['N']:
                    Seg = record[label_index[j] - S1:label_index[j] + S2]  # R峰的前0.4s和后0.5s
                    segment = resample(Seg, TAR_DIM, axis=0)  # 重采样到256
                    N_Seg.append(segment)

                if label[j] in AAMI_dic['S']:
                    Seg = record[label_index[j] - S1:label_index[j] + S2]
                    segment = resample(Seg, TAR_DIM, axis=0)
                    SVEB_Seg.append(segment)

                if label[j] in AAMI_dic['V']:
                    Seg = record[label_index[j] - S1:label_index[j] + S2]
                    segment = resample(Seg, TAR_DIM, axis=0)
                    VEB_Seg.append(segment)

                if label[j] in AAMI_dic['F']:
                    Seg = record[label_index[j] - S1:label_index[j] + S2]
                    segment = resample(Seg, TAR_DIM, axis=0)
                    F_Seg.append(segment)

                if label[j] in AAMI_dic['Q']:
                    Seg = record[label_index[j] - S1:label_index[j] + S2]
                    segment = resample(Seg, TAR_DIM, axis=0)
                    Q_Seg.append(segment)

    N_segement = np.array(N_Seg)
    SVEB_segement = np.array(SVEB_Seg)
    VEB_segement = np.array(VEB_Seg)
    F_segement = np.array(F_Seg)
    Q_segement = np.array(Q_Seg)

    label_N = np.zeros(N_segement.shape[0])
    label_SVEB = np.ones(SVEB_segement.shape[0])
    label_VEB = np.ones(VEB_segement.shape[0]) * 2
    label_F = np.ones(F_segement.shape[0]) * 3
    label_Q = np.ones(Q_segement.shape[0]) * 4

    Data = np.concatenate((N_segement, SVEB_segement, VEB_segement, F_segement, Q_segement), axis=0)
    Label = np.concatenate((label_N, label_SVEB, label_VEB, label_F, label_Q), axis=0)

    return Data, Label


# -----------------------心拍截取和保存---------------------
# 建议一次性截取和保存，不需要重复操作，下次训练和测试的时候，直接load
panth = './databases/mit-bih-arrhythmia-database-1.0.0/'
file = glob.glob(panth + '*.hea')
print(file)
Data, Label = heartbeat(file)
print(Data.shape)
print(Label.shape)
Data = np.save('Data', Data)
Label = np.save('Label', Label)
