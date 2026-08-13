# EEG 特征提取
import matplotlib.pyplot as plt
import pandas as pd
import csv
from collections import Counter
import numpy as np
from scipy import signal
import pywt
from scipy.fftpack import fft
import scipy.stats
from sklearn.preprocessing import MinMaxScaler
import math
import neurokit2 as nk
# from ..preprocess.eeg_preprocess import EEG_filter
 
def min_eeg(X, axis:int):
    '''
    Description: 求信号最小值
    -------------------------------
    Parameters:
    X: 输入EEG信号
    '''
    return np.min(X, axis=axis)
 
def max_eeg(X, axis:int):
    '''
    Description: 求信号最大值
    -------------------------------
    Parameters:
    X: 输入EEG信号
    '''
    return np.max(X, axis=axis)
 
def avg_eeg(X, axis:int):
    '''
    Description: 求信号平均值
    -------------------------------
    Parameters:
    X: 输入EEG信号
    '''
    return np.average(X, axis=axis)
 
def var_eeg(X, axis:int):
    '''
    Description: 求信号方差
    -------------------------------
    Parameters:
    X: 输入EEG信号
    '''
    return np.var(X, axis=axis)
 
def std_eeg(X, axis:int):
    '''
    Description: 求信号标准差
    -------------------------------
    Parameters:
    X: 输入EEG信号
    '''
    return np.std(X, axis=axis)
 
def mean_eeg(X, axis:int):
    '''
    Description: 求信号算数平均值
    -------------------------------
    Parameters:
    X: 输入EEG信号
    '''
    return np.mean(X, axis=axis)
 
def shannonentropy_eeg(X):
    '''
    Description: 求EEG信号香农熵
    -------------------------------
    Parameters:
    X: 输入EEG信号
    '''
    if len(X.shape) == 2:
        y = np.zeros((X.shape[1]))
        for i in range(X.shape[1]):
            y[i], _ = nk.entropy_shannon(X[:,i])
    elif len(X.shape) == 3:
        y = np.zeros((X.shape[0], X.shape[2]))
        for i in range(X.shape[0]):
            for j in range(X.shape[2]):
                y[i,j], _ = nk.entropy_shannon(X[i,:,j])
    else:
        print("only can calculate entropy for array of 2 or 3 dims, current dim = " + str(len(X.shape)))
    return y
 
def powerentropy_eeg(X):
    '''
    Description: 求EEG信号香农熵权
    -------------------------------
    Parameters:
    X: 输入EEG信号
    ''' 
    if len(X.shape) == 2:
        y = np.zeros((X.shape[1]))
        for i in range(X.shape[1]):
            y[i], _ = nk.entropy_power(X[:,i])
    elif len(X.shape) == 3:
        y = np.zeros((X.shape[0], X.shape[2]))
        for i in range(X.shape[0]):
            for j in range(X.shape[2]):
                y[i,j], _ = nk.entropy_power(X[i,:,j])
    else:
        print("only can calculate entropy for array of 2 or 3 dims, current dim = " + str(len(X.shape)))
    return y
 
def renyientropy_eeg(X):
    '''
    Description: 求EEG信号雷尼熵
    -------------------------------
    Parameters:
    X: 输入EEG信号
    '''  
    if len(X.shape) == 2:
        y = np.zero((X.shape[1]))
        for i in range(X.shape[1]):
            y[i], _ = nk.entropy_renyi(X[:,i])
    elif len(X.shape)  == 3:
        y = np.zeros((X.shape[0],X.shape[2]))
        for i in range(X.shape[0]):
            for j in range(X.shape[2]):
               #y[i,j], _ = nk.entropy_renyi(X[i,:,j],alpha=0)#雷尼熵变哈特里熵
               #y[i,j], _ = nk.entropy_renyi(X[i,:,j],alpha=1)#雷尼熵变香农熵
               y[i,j], _ = nk.entropy_renyi(X[i,:,j],alpha=2)#雷尼熵变碰撞熵
    else:
        print("only can calculate entropy for array of 2 or 3 dims, current dim = " + str(len(X.shape)))
    return y
 
def tsallisentropy_eeg(X):
    '''
    Description: 求EEG信号tsallis熵
    -------------------------------
    Parameters:
    X: 输入EEG信号
    ''' 
    if len(X.shape) == 2:
        y = np.zero((X.shape[1]))
        for i in range(X.shape[1]):
            y[i], _ = nk.entropy_tsallis(X[:,i])
    elif len(X.shape)  == 3:
        y = np.zeros((X.shape[0],X.shape[2]))
        for i in range(X.shape[0]):
            for j in range(X.shape[2]):
                y[i,j], _ = nk.entropy_tsallis(X[i,:,j],q=1)
    else:
        print("only can calculate entropy for array of 2 or 3 dims, current dim = " + str(len(X.shape)))
    return y
 
def teager_eeg(X):
    '''
    Description: 求EEG信号Teager能量
    -------------------------------
    Parameters:
    X: 输入EEG信号
    ''' 
    teager = np.zeros((X.shape))
    if len(X.shape) == 2:
        for i in range(1, X.shape[0]-1):
            for j in range(X.shape[1]):
                teager[i,j] = X[i,j] * X[i,j] - X[i-1,j] * X[i+1,j]    
    elif len(X.shape) == 3:
        for i in range(X.shape[0]):
            for j in range(1, X.shape[1]-1):
                for k in range(X.shape[2]):
                    teager[i,j,k] = X[i,j,k] * X[i,j,k] - X[i,j-1,k] * X[i,j+1,k]
    else:
        print("only can calculate teager for array of 2 or 3 dims, current dim = " + str(len(X.shape)))
    return teager
 
def mean_power(X):
    '''
    Description: 求EEG信号平均能量，为求teager平均能量
    -------------------------------
    Parameters:
    X: 输入EEG信号
    '''  
    if len(X.shape) == 2:
        y = np.zeros((X.shape[1]))
        for i in range(X.shape[1]):
            y[i] = np.mean(X[:,i] ** 2)
    elif len(X.shape) == 3:
        y = np.zeros((X.shape[0], X.shape[2]))
        for i in range(X.shape[0]):
            for j in range(X.shape[2]):
                y[i,j] = np.mean(X[i,:,j] ** 2)
    else:
        print("only can calculate entropy for array of 2 or 3 dims, current dim = " + str(len(X.shape)))     
    return y
 
def avg_teager_power(X):
    '''
    Description: 求EEG信号teager平均能量
    -------------------------------
    Parameters:
    X: 输入EEG信号
    ''' 
    teager = teager_eeg(X)
    y = mean_power(teager)
    return y
 
def avg_power(X):
    '''
    Description: 求EEG信号平均能量
    -------------------------------
    Parameters:
    X: 输入EEG信号
    ''' 
    y = mean_power(X)
    return y
 
def first_diff(X):
    '''
    Description: 求EEG信号一阶差
    -------------------------------
    Parameters:
    X: 输入EEG信号
    ''' 
    if len(X.shape) == 2:
        diff = np.diff(X, n=1, axis=0)
        y = np.pad(diff, ((0,1),(0,0)), "constant", constant_values=0)
    elif len(X.shape) == 3:
        diff = np.diff(X, n=1, axis=1)
        y = np.pad(diff, ((0,0),(0,1),(0,0)), "constant", constant_values=0)
    return y
 
def first_diff_avg(X, axis:int):
    '''
    Description: 求EEG信号一阶差均值
    -------------------------------
    Parameters:
    X: 输入EEG信号
    ''' 
    diff1 = first_diff(X)
    y = np.mean(diff1, axis=axis)
    return y
 
def first_diff_absavg(X, axis:int):
    '''
    Description: 求EEG信号一阶差均值的绝对值
    -------------------------------
    Parameters:
    X: 输入EEG信号
    ''' 
    diff1 = first_diff(X)
    y = np.mean(np.abs(diff1), axis=axis)
    return y
 
def first_diff_median(X, axis:int):
    '''
    Description: 求EEG信号一阶差中位数
    -------------------------------
    Parameters:
    X: 输入EEG信号
    ''' 
    diff1 = first_diff(X)
    y = np.median(diff1, axis=axis)
    return y
 
def first_diff_absmedian(X, axis:int):
    '''
    Description: 求EEG信号一阶差中位数绝对值
    -------------------------------
    Parameters:
    X: 输入EEG信号
    '''    
    diff1 = first_diff(X)
    y = np.median(np.abs(diff1), axis=axis)
    return y
 
def first_diff_sumabs(X, axis:int):
    '''
    Description: 求EEG信号一阶差绝对值的和
    -------------------------------
    Parameters:
    X: 输入EEG信号
    ''' 
    diff1 = first_diff(X)
    y = np.sum(np.abs(diff1), axis=axis)
    return y
 
def second_diff(X):
    '''
    Description: 求EEG信号二阶差
    -------------------------------
    Parameters:
    X: 输入EEG信号
    ''' 
    if len(X.shape) == 2:
        diff = np.diff(X, n=2, axis=0)
        y = np.pad(diff, ((0,2),(0,0)), "constant", constant_values=0)
    elif len(X.shape) == 3:
        diff = np.diff(X, n=2, axis=1)
        y = np.pad(diff, ((0,0),(0,2),(0,0)), "constant", constant_values=0)
    return y
 
def norm_first_diff(X):
    '''
    Description: 求EEG信号归一化一阶差
    -------------------------------
    Parameters:
    X: 输入EEG信号
    ''' 
    diff = first_diff(X)
    if len(X.shape) == 2:
        y = MinMaxScaler().fit_transform(diff)
    elif len(X.shape) == 3:
        y = np.zeros((diff.shape))
        for i in range(diff.shape[0]):
            y[i,:,:] = MinMaxScaler().fit_transform(diff[i,:,:])
    return y
 
def norm_second_diff(X):
    '''
    Description: 求EEG信号归一化二阶差
    -------------------------------
    Parameters:
    X: 输入EEG信号
    ''' 
    diff = second_diff(X)
    if len(X.shape) == 2:
        y = MinMaxScaler().fit_transform(diff)
    elif len(X.shape) == 3:
        y = np.zeros((diff.shape))
        for i in range(diff.shape[0]):
            y[i,:,:] = MinMaxScaler().fit_transform(diff[i,:,:])
    return y
 
def kurt(X):
    '''
    Description: 求EEG信号峰度
    -------------------------------
    Parameters:
    X: 输入EEG信号
    ''' 
    if len(X.shape) == 2:
        y = np.zeros(X.shape[1])
        for i in range(X.shape[1]):
            s = pd.Series(X[:,i])
            y[i] = s.kurt()
    elif len(X.shape) == 3:
        y = np.zeros((X.shape[0], X.shape[2]))
        for i in range(X.shape[0]):
            for j in range(X.shape[2]):
                s = pd.Series(X[i,:,j])
                y[i,j] = s.kurt()
    return y
 
def skew(X):
    '''
    Description: 求EEG信号偏度
    -------------------------------
    Parameters:
    X: 输入EEG信号
    ''' 
    if len(X.shape) == 2:
        y = np.zeros(X.shape[1])
        for i in range(X.shape[1]):
            s = pd.Series(X[:,i])
            y[i] = s.skew()
    elif len(X.shape) == 3:
        y = np.zeros((X.shape[0], X.shape[2]))
        for i in range(X.shape[0]):
            for j in range(X.shape[2]):
                s = pd.Series(X[i,:,j])
                y[i,j] = s.skew()
    return y
 
def hjorth_activity(X):
    '''
    Description: 求EEG信号平均能量
    -------------------------------
    Parameters:
    X: 输入EEG信号
    ''' 
    if len(X.shape) == 2:
        var = np.var(X, axis=0)
        y = var ** 2
    elif len(X.shape) == 3:
        var = np.var(X, axis=1)
        y = var ** 2
    return y    
 
def hjorth_mobility(X):
    '''
    Description: 求EEG信号Hjorth活动性
    -------------------------------
    Parameters:
    X: 输入EEG信号
    ''' 
    if len(X.shape) == 2:
        var = np.var(X, axis=0)
        diff = np.diff(var, n=1, axis=0)
        pad_diff = np.pad(diff, ((0,1)), "constant", constant_values=0)
        y = pad_diff / var
    elif len(X.shape) == 3:
        var = np.var(X, axis=1)
        diff = np.diff(var, n=1, axis=0)
        pad_diff = np.pad(diff, ((0,1),(0,0)), "constant", constant_values=0)
        y = pad_diff // var
    return y 
 
def hjorth_complexity(X):
    '''
    Description: 求EEG信号Hjorth复杂性
    -------------------------------
    Parameters:
    X: 输入EEG信号
    ''' 
    if len(X.shape) == 2:
        var = np.var(X, axis=0)
        diff = np.diff(var, n=1, axis=0)
        pad_diff = np.pad(diff, ((0,1)), "constant", constant_values=0)
        diff2 = np.diff(var, n=2, axis=0)
        pad_diff2 = np.pad(diff2, ((0,2)), "constant", constant_values=0)
        y = pad_diff2 * var // (pad_diff ** 2)
    elif len(X.shape) == 3:
        var = np.var(X, axis=1)
        diff = np.diff(var, n=1, axis=0)
        pad_diff = np.pad(diff, ((0,1),(0,0)), "constant", constant_values=0)
        diff2 = np.diff(var, n=2, axis=0)
        pad_diff2 = np.pad(diff2, ((0,2),(0,0)), "constant", constant_values=0)
        y = pad_diff2 * var // (pad_diff ** 2)
    return y
 
def fft_eeg(X):
    '''
    Description: 对EEG信号进行快速傅里叶变换
    -------------------------------
    Parameters:
    X: 输入EEG信号
    ''' 
    y = np.zeros((X.shape))
    if len(X.shape) == 2:
        for i in range(X.shape[1]):
            y[:,i] = fft(X[:,i])
    elif len(X.shape) == 3:
        for i in range(X.shape[0]):
            for j in range(X.shape[2]):
                y[i,:,j] = fft(X[i,:,j])           
    return y
 
def fft_meancoef(X):
    '''
    Description: 求傅里叶变换平均系数
    -------------------------------
    Parameters:
    X: 输入EEG信号
    ''' 
    amp = np.abs(fft_eeg(X))
    if len(X.shape) == 2:
        y = np.mean(amp, axis=0)
    elif len(X.shape) == 3:
        y = np.mean(amp, axis=1) 
    return y
 
def fft_fp(X):
    '''
    Description: 求傅里叶变换绝对值
    -------------------------------
    Parameters:
    X: 输入EEG信号
    ''' 
    return np.abs(fft_eeg(X))
 
def fft_maxF(X):
    '''
    Description: 求傅里叶变换最大值
    -------------------------------
    Parameters:
    X: 输入EEG信号
    ''' 
    fp = fft_fp(X)
    if len(X.shape) == 2:
        y = np.max(fp, axis=0)
    elif len(X.shape) == 3:
        y = np.max(fp, axis=1) 
    return y
 
def fft_medianF(X):
    '''
    Description: 求傅里叶变换中位数
    -------------------------------
    Parameters:
    X: 输入EEG信号
    ''' 
    fp = fft_fp(X)
    if len(X.shape) == 2:
        y = np.median(fp, axis=0)
    elif len(X.shape) == 3:
        y = np.median(fp, axis=1) 
    return y
 
def fft_peak(X):
    '''
    Description: 求傅里叶变换峰值
    -------------------------------
    Parameters:
    X: 输入EEG信号
    ''' 
    fp = fft_fp(X)
    if len(X.shape) == 2:
        y = np.zeros(X.shape[1])
        for i in range(X.shape[1]):
            info = nk.signal_findpeaks(fp[:,i])
            heights = info["Height"]
            if heights.size == 0:
                y[i] = 0
            else:
                y[i] = max(heights)
    elif len(X.shape) == 3:
        y = np.zeros((X.shape[0], X.shape[2]))
        for i in range(X.shape[0]):
            for j in range(X.shape[2]):
                info = nk.signal_findpeaks(fp[i,:,j])
                heights = info["Height"]
                if heights.size == 0:
                    y[i,j] = 0
                else:
                    y[i,j] = max(heights)
    return y
    
def wt_eeg(X, order:str):
    '''
    Description: 对EEG信号进行小波变换
    -------------------------------
    Parameters:
    X: 输入EEG信号
    ''' 
    if len(X.shape) == 2:
        y = np.zeros((2, X.shape[0], X.shape[1]))
        for i in range(X.shape[1]):
            cA, cD = pywt.dwt(X[:,i], order)
            ya = pywt.idwt(cA, None, order, 'smooth')
            yd = pywt.idwt(None, cD, order, 'smooth')
            y[0,:,i] = ya
            y[1,:,i] = yd
    elif len(X.shape) == 3:
        y = np.zeros((2, X.shape[0], X.shape[1], X.shape[2]))
        for i in range(X.shape[0]):
            for j in range(X.shape[2]):
                cA, cD = pywt.dwt(X[i,:,j], order)
                ya = pywt.idwt(cA, None, order, 'smooth')
                yd = pywt.idwt(None, cD, order, 'smooth')
                y[0,i,:,j] = ya
                y[1,i,:,j] = yd               
    return y
 
def wt_absmean(X, order:str):
    '''
    Description: 求小波绝对均值
    -------------------------------
    Parameters:
    X: 输入EEG信号
    ''' 
    wt = wt_eeg(X, order)
    if len(X.shape) == 2:
        y = np.abs(np.mean(wt, axis=1))
    elif len(X.shape) == 3:
        y = np.abs(np.mean(wt, axis=2))
    return y    
 
def wt_std(X, order:str):
    '''
    Description: 求小波标准差
    -------------------------------
    Parameters:
    X: 输入EEG信号
    ''' 
    wt = wt_eeg(X, order)
    if len(X.shape) == 2:
        y = np.std(wt, axis=1)
    elif len(X.shape) == 3:
        y = np.std(wt, axis=2)
    return y
 
def wt_var(X, order:str):
    '''
    Description: 求小波方差
    -------------------------------
    Parameters:
    X: 输入EEG信号
    ''' 
    wt = wt_eeg(X, order)
    if len(X.shape) == 2:
        y = np.var(wt, axis=1)
    elif len(X.shape) == 3:
        y = np.var(wt, axis=2)
    return y
 
def fft_autocor(X):
    '''
    Description: 计算傅里叶变换的自相关性
    -------------------------------
    Parameters:
    X: 输入EEG信号
    ''' 
    fp = fft_fp(X)
    r, info = nk.signal_autocor(fp, method='auto')
    return r
 
def centroid(X):
    '''
    Description: 计算EEG信号中心坐标
    -------------------------------
    Parameters:
    X: 输入EEG信号
    ''' 
    y = scipy.ndimage.center_of_mass(X)
    return y
 
def peak_distance(X):
    '''
    Description: 计算EEG信号波峰波谷之间距离
    -------------------------------
    Parameters:
    X: 输入EEG信号
    ''' 
    if len(X.shape) == 2:
        y = np.zeros(X.shape[1])
        for i in range(X.shape[1]):
            info = nk.signal_findpeaks(X[:,i])
            widths = info["Width"]
            if widths.size == 0:
                y[i] = 0
            else:
                y[i] = max(widths)
    elif len(X.shape) == 3:
        y = np.zeros((X.shape[0], X.shape[2]))
        for i in range(X.shape[0]):
            for j in range(X.shape[2]):
                info = nk.signal_findpeaks(X[i,:,j])
                widths = info["Width"]
                if widths.size == 0:
                    y[i,j] = 0
                else:
                    y[i,j] = max(widths)
    return y
 
def peak_maxnum(X):
    '''
    Description: 计算EEG信号最大峰值个数
    -------------------------------
    Parameters:
    X: 输入EEG信号
    ''' 
    if len(X.shape) == 2:
        y = np.zeros(X.shape[1])
        for i in range(X.shape[1]):
            info = nk.signal_findpeaks(X[:,i])
            peaks = info["Peaks"]
            if peaks.size == 0:
                y[i] = 0
            else:
                y[i] = len(peaks)
    elif len(X.shape) == 3:
        y = np.zeros((X.shape[0], X.shape[2]))
        for i in range(X.shape[0]):
            for j in range(X.shape[2]):
                info = nk.signal_findpeaks(X[i,:,j])
                peaks = info["Peaks"]
                if peaks.size == 0:
                    y[i,j] = 0
                else:
                    y[i,j] = len(peaks)
    return y
 
def peak_minnum(X):
    '''
    Description: 计算EEG信号最小峰值个数
    -------------------------------
    Parameters:
    X: 输入EEG信号
    ''' 
    Y = -X
    if len(X.shape) == 2:
        y = np.zeros(X.shape[1])
        for i in range(X.shape[1]):
            info = nk.signal_findpeaks(Y[:,i])
            peaks = info["Peaks"]
            if peaks.size == 0:
                y[i] = 0
            else:
                y[i] = len(peaks)
    elif len(X.shape) == 3:
        y = np.zeros((X.shape[0], X.shape[2]))
        for i in range(X.shape[0]):
            for j in range(X.shape[2]):
                info = nk.signal_findpeaks(Y[i,:,j])
                peaks = info["Peaks"]
                if peaks.size == 0:
                    y[i,j] = 0
                else:
                    y[i,j] = len(peaks)
    return y
 
def zc_eeg(X):
    '''
    Description: 计算EEG信号跨零率
    -------------------------------
    Parameters:
    X: 输入EEG信号
    ''' 
    if len(X.shape) == 2:
        y = np.zeros((X.shape))
        for i in range(X.shape[1]):
            zc = nk.signal_zerocrossings(X[:,i])
            for j in range(len(zc)):
                y[i,j] = zc[j]
    elif len(X.shape) == 3:
        y = np.zeros((X.shape))
        for i in range(X.shape[0]):
            for j in range(X.shape[2]):
                zc = nk.signal_zerocrossings(X[i,:,j])
                for k in range(len(zc)):
                    y[i,k,j] = zc[k]
    else:
        print("only can calculate zero crossing for 2 and 3 dims, current dim = " + str(len(X.shape)))
        y = "" 
    return y
 
def filter(X, sample_rate: int, low: float = 0.5, high: float = None, method: str = 'butterworth', order: int = 2):
    """
    Description: nk滤波器实现
    -------------------------------
    Parameters:
    X: 输入信号array
    sample_rate: 采样率
    low: 低频带
    high: 高频带
    method: 滤波器的类型，支持："butterworth", "fir", "bessel" or "savgol"
    order: 滤波器阶数
    """
    # 读入数据array，经过滤波器后，输出数据array
    if len(X.shape) == 1 or 2:
        y = nk.signal_filter(X, sampling_rate=sample_rate, lowcut=low, highcut=high, method=method, order=order)
    elif len(X.shape) == 3:
        y = np.zeros((X.shape))
        for i in range(X.shape[0]):
            y[i,:,:] = nk.signal_filter(X[i,:,:], sampling_rate=sample_rate, lowcut=low, highcut=high, method=method, order=order)
    else:
        print("only support dim less or equal to 3, but current dim = " + str(len(X.shape)))
    return y
 
def EEG_filter(X, sample_rate: int, method: str = 'butterworth', order: int = 2):
    """
    Description: 对EEG信号通过带通滤波提取delta，theta，alpha，beta波
    -------------------------------
    Parameters:
    X: 输入信号array
    sample_rate: 输入信号采样率
    order: 滤波器阶数  
    """
    # 读入数据array，输出四类波的array
    delta = filter(X, sample_rate=sample_rate, low=1, high=3, method=method, order=order)
    theta = filter(X, sample_rate=sample_rate, low=4, high=7, method=method, order=order)
    alpha = filter(X, sample_rate=sample_rate, low=8, high=13, method=method, order=order)
    beta = filter(X, sample_rate=sample_rate, low=14, high=30, method=method, order=order)
    gamma = filter(X, sample_rate=sample_rate, low=31, method=method, order=order)
 
    return delta, theta, alpha, beta, gamma
 
def sum_power(X, sample_rate: int):
    '''
    Description: 计算EEG信号各个波频总能量(绝对能量)
    -------------------------------
    Parameters:
    X: 输入EEG信号
    sample_rate = EEG信号采样率
    ''' 
    delta, theta, alpha, beta, gamma = EEG_filter(X, sample_rate=sample_rate)
    X = np.array([delta, theta, alpha, beta])
    sumpower = np.sum(np.sum(X, axis=0), axis=1)
    return sumpower
 
def delta_power(X, sample_rate: int):
    '''
    Description: 计算EEG信号delta波频绝对能量
    -------------------------------
    Parameters:
    X: 输入EEG信号
    sample_rate = EEG信号采样率
    ''' 
    delta, theta, alpha, beta, gamma = EEG_filter(X, sample_rate=sample_rate)
    deltapower = np.sum(delta, axis=1)
    return deltapower
 
def theta_power(X, sample_rate: int):
    '''
    Description: 计算EEG信号theta波频绝对能量
    -------------------------------
    Parameters:
    X: 输入EEG信号
    sample_rate = EEG信号采样率
    ''' 
    delta, theta, alpha, beta, gamma = EEG_filter(X, sample_rate=sample_rate)
    thetapower = np.sum(theta, axis=1)
    return thetapower
 
def alpha_power(X, sample_rate: int):
    '''
    Description: 计算EEG信号alpha波频绝对能量
    -------------------------------
    Parameters:
    X: 输入EEG信号
    sample_rate = EEG信号采样率
    ''' 
    delta, theta, alpha, beta, gamma = EEG_filter(X, sample_rate=sample_rate)
    alphapower = np.sum(alpha, axis=1)
    return alphapower
 
def beta_power(X, sample_rate: int):
    '''
    Description: 计算EEG信号beta波频绝对能量
    -------------------------------
    Parameters:
    X: 输入EEG信号
    sample_rate = EEG信号采样率
    ''' 
    delta, theta, alpha, beta, gamma = EEG_filter(X, sample_rate=sample_rate)
    betapower = np.sum(beta, axis=1)
    return betapower
 
def delta_relativepower(X, sample_rate: int):
    '''
    Description: 计算EEG信号delta波频相对能量
    -------------------------------
    Parameters:
    X: 输入EEG信号
    sample_rate = EEG信号采样率
    ''' 
    delta, theta, alpha, beta, gamma = EEG_filter(X, sample_rate=sample_rate)
    sumpower = sum_power(X, sample_rate=sample_rate)
    deltapower = np.sum(delta, axis=1) / sumpower
    return deltapower
 
def theta_relativepower(X, sample_rate: int):
    '''
    Description: 计算EEG信号theta波频相对能量
    -------------------------------
    Parameters:
    X: 输入EEG信号
    sample_rate = EEG信号采样率
    ''' 
    delta, theta, alpha, beta, gamma = EEG_filter(X, sample_rate=sample_rate)
    sumpower = sum_power(X, sample_rate=sample_rate)
    thetapower = np.sum(theta, axis=1) / sumpower
    return thetapower
 
def alpha_relativepower(X, sample_rate: int):
    '''
    Description: 计算EEG信号alpha波频相对能量
    -------------------------------
    Parameters:
    X: 输入EEG信号
    sample_rate = EEG信号采样率
    ''' 
    delta, theta, alpha, beta, gamma = EEG_filter(X, sample_rate=sample_rate)
    sumpower = sum_power(X, sample_rate=sample_rate)
    alphapower = np.sum(alpha, axis=1) / sumpower
    return alphapower
 
def beta_relativepower(X, sample_rate: int):
    '''
    Description: 计算EEG信号beta波频相对能量
    -------------------------------
    Parameters:
    X: 输入EEG信号
    sample_rate = EEG信号采样率
    ''' 
    delta, theta, alpha, beta, gamma = EEG_filter(X, sample_rate=sample_rate)
    sumpower = sum_power(X, sample_rate=sample_rate)
    betapower = np.sum(beta, axis=1) / sumpower
    return betapower
 
def relativepower1(X, sample_rate: int):
    '''
    Description: 计算EEG信号(theta+alpha)/beta相对能量
    -------------------------------
    Parameters:
    X: 输入EEG信号
    sample_rate = EEG信号采样率
    ''' 
    delta, theta, alpha, beta, gamma = EEG_filter(X, sample_rate=sample_rate)
    x = np.array([theta, alpha])
    sumpower = np.sum(np.sum(x, axis=0), axis=1)
    relativepower = sumpower / np.sum(beta,axis=1)
    return relativepower
 
def relativepower2(X, sample_rate: int):
    '''
    Description: 计算EEG信号alpha/beta相对能量
    -------------------------------
    Parameters:
    X: 输入EEG信号
    sample_rate = EEG信号采样率
    ''' 
    delta, theta, alpha, beta, gamma = EEG_filter(X, sample_rate=sample_rate)
    relativepower = np.sum(alpha, axis=1) / np.sum(beta, axis=1)
    return relativepower
 
def relativepower3(X, sample_rate: int):
    '''
    Description: 计算EEG信号(theta+alpha)/(beta+alpha)相对能量
    -------------------------------
    Parameters:
    X: 输入EEG信号
    sample_rate = EEG信号采样率
    ''' 
    delta, theta, alpha, beta, gamma = EEG_filter(X, sample_rate=sample_rate)
    x1 = np.array([theta, alpha])
    sumpower1 = np.sum(np.sum(x1, axis=0), axis=1)
    x2 = np.array([beta, alpha])
    sumpower2 = np.sum(np.sum(x2, axis=0), axis=1)
    relativepower = sumpower1 / sumpower2
    return relativepower
 
def relativepower4(X, sample_rate: int):
    '''
    Description: 计算EEG信号theta/beta相对能量
    -------------------------------
    Parameters:
    X: 输入EEG信号
    sample_rate = EEG信号采样率
    ''' 
    delta, theta, alpha, beta, gamma = EEG_filter(X, sample_rate=sample_rate)
    relativepower = np.sum(theta, axis=1) / np.sum(beta, axis=1)
    return relativepower
    
def loadEEGCSV(data_path: str, window: float, frame: float, sample_rate: int, channels: int):
    """
    Description: 加载csv文件
    -------------------------------
    Parameters:
    data_path: 输入的csv文件
    window:  窗的大小（单位：s）
    frame:  帧的大小，即窗移的大小（单位：s）
    sample_rate:  采样率的大小（单位：Hz）
    """
    #load feature data
    df = pd.read_csv(data_path)
    FEATURE_NUM = int(window * sample_rate)
    FRAME_NUM = int(frame * sample_rate)    
    data = df.iloc[:,1:].values
    if channels != data.shape[1]:
        print("Error in channel num = " + str(data.shape[1]))
    j = 0
    X = np.zeros((int(data.shape[0]/FRAME_NUM), FEATURE_NUM, channels))
    for i in range(0,(data.shape[0] - FEATURE_NUM),FRAME_NUM):
        X[j,:,:] = data[i:i+FEATURE_NUM,:]
        j = j + 1
     
    return X
 
if __name__ == '__main__':  
    data_path = "D:\work\EEG5_RawData(RelativeRecording).csv"
    X = loadEEGCSV(data_path,window=1.0, frame=0.5, sample_rate=256, channels=16)
    print(X.shape) 
    # min_matrix = min_eeg(X, axis=1)
    # max_matrix = max_eeg(X, axis=1)
    # avg_matrix = avg_eeg(X, axis=1)
    # var_matrix = var_eeg(X, axis=1)
    # std_matrix = std_eeg(X, axis=1)
    # mean_matrix = mean_eeg(X, axis=1)
    # teager = teager_eeg(X)
    # shannon = shannonentropy_eeg(X)
    # teageravgpower = avg_teager_power(X)
    # avgpower = avg_power(X)
    # firdiff = first_diff(X)
    # secdiff = second_diff(X)
    # norm1diff = norm_first_diff(X)
    # norm2diff = norm_second_diff(X)
    # fengdu = kurt(X)
    # piandu = skew(X)
    # hj_act = hjorth_activity(X)
    # hj_mob = hjorth_mobility(X)
    # hj_comp = hjorth_complexity(X)
    # ffteeg = fft_eeg(X)
    # fft_mean = fft_meancoef(X)
    # wteeg = wt_eeg(X, order='db5')
    # wtabsmean = wt_absmean(X, order='db5')
    # wtstd = wt_std(X, order='db5')
    # wtvar = wt_var(X, order='db5')
    # fp = fft_fp(X)
    # maxf = fft_maxF(X)
    # medianf = fft_medianF(X)
    # fftpeak = fft_peak(X)
    # fftauto = fft_autocor(X)
    # cent = centroid(X)
    # avgdiff1 = first_diff_avg(X, axis=1)
    # avgabsdiff1 = first_diff_absavg(X, axis=1)
    # peakmaxnum = peak_maxnum(X)
    # zc = zc_eeg(X)
    # renyentro = renyientropy_eeg(X)
    # tsalentro = tsallisentropy_eeg(X)
    # powerentro = powerentropy_eeg(X)
    # sumpower = sum_power(X, sample_rate=256)
    # delpow = delta_power(X, sample_rate=256)
    # delrelapow = delta_relativepower(X, sample_rate=256)
    rela1 = relativepower1(X, sample_rate=256)
    print(rela1.shape)