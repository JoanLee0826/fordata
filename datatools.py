#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as ss
import matplotlib.pyplot as plt # 
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus']=False
get_ipython().run_line_magic('matplotlib', 'inline')
from itertools import combinations # 自由组合


# In[3]:


# 一些辅助的函数
def cm_plot(y, yp): # 混淆矩阵
    """
    y:实际分类
    yp: 预测分类
    """
    from sklearn.metrics import confusion_matrix
    import matplotlib.pyplot as plt
    import seaborn as sns
    cm = confusion_matrix(y, yp)
    sns.heatmap(cm, annot=True,fmt='d',cmap=plt.cm.Greens)
    plt.xlabel("Real Label")
    plt.ylabel('Predicted Label')
    return plt

def radar(df): # 雷达图
    """
    输入df
    df.index = "分类"
    df.columns = "要素" 
    """
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    result = pd.concat([df, df.iloc[:,0]], axis=1)  # 闭合的 dataframe
    kind = list(result.index)
    
    angles = np.linspace(0, 2*np.pi, len(df.columns), endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))  # 闭合的角度
    
    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(111, polar=True) # 开启极坐标
    
    for each in kind:
        ax.plot(angles, result.loc[each,:])
        ax.fill(angles, result.loc[each,:], alpha=0.1,label=str(each)) # 填充
    
    plt.legend(loc='lower right')
                
    ax.set_thetagrids(angles * 180 / np.pi, df.columns,fmt='large') #设置极坐标角度网格线显示
#     plt.title('rader-distribute')
    return plt


# In[6]:


radar(df).savefig('pic.png',dpi=200)


# In[7]:


def minmax(ser): 
    ser = ser.agg(lambda x : (x-x.min())/(x.max()-x.min()))
    return ser.astype(np.float64)
def zscore(ser):
    ser = ser.agg(lambda x : (x-x.mean()) / x.std())
    return ser.astype(np.float64)


# In[8]:


def get_dum(data):
    """
    输入DataFrame 转化为哑变量矩阵的DataFrame
    """
    df = pd.DataFrame()
    for i in range(len(data)):
        for j in range(len(data.columns)):
            if not pd.isna(self.data.at[i, j]):
                df.at[i, data.at[i, j]] = 1
    return df.fillna(0)


# In[9]:


def get_pca(data_mat, n_component=1000000): # 传统的PCA方法
    from scipy import linalg 
    mean_vals = np.mean(data_mat, axis=0)
    mid_mat = data_mat - mean_vals
    cov_mat = np.cov(mid_mat, rowvar=False)
    eig_vals, eig_vects = linalg.eig(np.mat(cov_mat))
    eig_val_index = np.argsort(-eig_vals)
    eig_val_index = eig_val_index[:n_component]
    eig_vects = eig_vects[:, eig_val_index]
    low_dim_mat = np.dot(mid_mat, eig_vects)
    # ret_mat = np.dot(low_dim_mat,eig_vects.T)
    return low_dim_mat, eig_vals


# In[10]:


def pca(data_mat, topNfeat=1000000):
    from scipy import linalg
    mean_vals = np.mean(data_mat, axis=0)
    mid_mat = data_mat - mean_vals
    cov_mat = np.cov(mid_mat, rowvar=False)
    eig_vals, eig_vects = linalg.eig(np.mat(cov_mat))
    eig_val_index = np.argsort(eig_vals)
    eig_val_index = eig_val_index[:-(topNfeat + 1):-1]
    eig_vects = eig_vects[:, eig_val_index]
    low_dim_mat = np.dot(mid_mat, eig_vects)
    # ret_mat = np.dot(low_dim_mat,eig_vects.T)
    return low_dim_mat, eig_vals


# In[2]:


from itertools import combinations


class Apriori:
    """
    商品推荐 输入用户购买记录，可以进行DataFrame的序列文件
    输出"商品_关联商品"的支持度，可信度
    """

    def __init__(self):

        self.df = pd.DataFrame()
        self.lst_sum = []
    
    def get_pre(self, data):
        self.data = data
        if not isinstance(data, pd.DataFrame):
            try:
                self.data = pd.DataFrame(self.data)
            except Exception as e:
                print(e, "请输入DataFrame类型的数据")
                
    def get_dum(self):
        for i in range(len(self.data)):
            for j in range(len(self.data.columns)):
                if  not pd.isna(self.data.at[i, j]) :
                    self.df.at[i, self.data.at[i, j]] = 1

        self.df.fillna(0, inplace=True)
        return self.df

    def get_com(self, n=3):
        lst = self.df.columns
        for i in range(1, n + 1):
            for each in combinations(lst, i):
                if each:
                    self.lst_sum.append(list(each))
        return self.lst_sum

    def get_sup(self, sup, con):
        res = pd.DataFrame()
        for each in self.lst_sum:
            data_tem = self.df[each].cumprod(axis=1)
            p_sup = data_tem.iloc[:, -1].sum() / len(self.df[each])

            if p_sup > sup:
                if len(each) == 1:
                    p_con = None
                    print(each, p_sup, p_con)
                    res.loc["_".join(each), 'sup'] = p_sup

                else:
                    #  np.seterr(invalid='ignore')
                    if data_tem.iloc[:, -2].sum():
                        p_con = data_tem.iloc[:, -1].sum() / data_tem.iloc[:, -2].sum()

                        if p_con > con or p_con is None:
                            print(each, p_sup, p_con)
                            res.loc["_".join(each), 'sup'] = p_sup
                            res.loc["_".join(each), 'con'] = p_con
        return res

    def fit(self, data, sup=0.05, con=0.05):
        self.get_pre(data=data)
        self.get_dum()
        self.get_com()
        res = self.get_sup(sup=sup, con=con)
        return res


# In[ ]:




