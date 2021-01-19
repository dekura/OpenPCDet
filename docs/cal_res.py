'''
Author: Guojin Chen @ CUHK-CSE
Homepage: https://dekura.github.io/
Date: 2021-01-18 10:28:41
LastEditTime: 2021-01-19 15:07:01
Contact: cgjhaha@qq.com
Description: calculate the results.
'''
import numpy as np

# the first version
# accu = np.array([67.6, 72.4, 82.6])
# fa = np.array([15, 39, 264])
# t = np.array([1.5, 8.0, 6.5])


accu = np.array([83.1, 88.4, 100.0])
fa = np.array([23, 69, 284])
t = np.array([1.6, 8.2, 5.5])
print('accu: {}'.format(accu))
print('fa: {}'.format(fa))
print('t: {}'.format(t))


m_accu = np.mean(accu)
m_fa = np.mean(fa)
m_t = np.mean(t)

rcnn = np.array([95.8, 84, 6])
our = np.array([m_accu, m_fa, m_t])/rcnn

print('m_accu: {}'.format(m_accu))
print('m_fa: {}'.format(m_fa))
print('m_t: {}'.format(m_t))
print('per: {}'.format(our))
