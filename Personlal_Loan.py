
# coding: utf-8

# In[4]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as ss
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set()  # 重置sns的默认设置
sns.set_style("whitegrid") # 设置seaborn的主题样式
# plt.rcParams['font.sans-serif'] = ['SimHei'] # 中文显示 注：seaborn不推荐中文显示
plt.rcParams['axes.unicode_minus'] = False # 正确显示"负号" 


# In[5]:


data = pd.read_excel('Personal_Loan.xlsx','Data')  # 读取文件"P_L.xlsx" 中的"Data" 表
data.head() 


# In[6]:


data.info() # 数据概况


# In[7]:


data.isnull().sum() # 没有空数据 


# In[8]:


data.describe().T  #  基本统计：计数 均值 标准差 最小最大值 四分位数


# - 简要分析数据各标签意义
#     - 二分类 (0/1)
#         - Personal Loan 是否接受贷款 (目标标签)
#         - Securities Account 是否有安全账号
#         - CD account 是否有存款证书
#         - Online 是否开通网银
#         - CreditCard 是否开通信用卡
#     - 定距分类 Interval :变量值可以比较大小  差值有意义
#         - Age 年龄
#         - Experience 就职时间
#         - Income 收入
#         - CCAvg 每月的信用卡消费
#         - Mortgage 房子可抵押款
#     - 定序分类 Ordinal：差值无明确意义 （例如：教育程度,有高低之分 没有高多少之分）
#         - Family 家庭成员数
#         - Education 受教育程度
#     - 定类变量 Norminal (数字大小没有意义)
#         - ID 区分用户
#         - ZIP Code 美国的邮政编码

# In[7]:


len(data[data.Experience < 0]) 


# - 数据目标：
#     - 分析"客户是否接收贷款的二值分类问题"，通过建模，得到"数据特征"--"数据标签"的关系。
# - 数据预处理：
#     1. 异常数据识别：
#         - 观察到，Experience 最小值为-3 有52条记录小于0，(认为) '-2' 代表2年后入职, 不予处理。
#     2. 定序数据处理：
#         - 定序数据通常需要进行**独热化** 转化为 只包含0,1的矩阵形式
#     3. 规范化处理：
#         - 减弱量纲对数据的影响，进行最小最大标准化， 或者Z分数标准化。

# In[15]:


def minmax(ser):
    """
    输入series 进行最小最大化
    """
    ser = (ser - ser.min())/ (ser.max() - ser.min())
    return ser.astype(np.float64)
def zscore(ser):
    """
    输入Series 进行Z分数标准化
    """
    ser = (ser - ser.min())/ (ser.std())
    return ser.astype(np.float64)


# In[21]:


data.drop(['ID','ZIP Code'],inplace=True, axis=1)


# In[49]:


corr = data.corr()
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True #  生成一个上三角阵 作为热力图的遮罩
plt.figure(figsize=(10,8))
sns.set_context(context='notebook')
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f')
plt.xticks(rotation=90)


# In[22]:


data2 = pd.get_dummies(data, columns=['Family','Education'])
data2.head()


# In[23]:


for each in ['Age','Experience','Income','CCAvg']:
    data2[each] = zscore(data2[each])
data2.head()


# In[52]:


sns.set(style='whitegrid', color_codes=True)
plt.figure(figsize=(10,8.2))
sns.set_context(context='notebook')
sns.heatmap(corr,mask=corr.abs()<0.2, annot=True, fmt='.2f',linewidths=0.01, linecolor='black') # 过滤掉相关性绝对值小于0.2的部分
plt.tight_layout()


# In[27]:


data2.columns


# In[28]:


label_v = data2['Personal Loan'].values 


# In[29]:


feature_data = data2.drop('Personal Loan', axis=1)
feature_v = feature_data.values
feature_names = feature_data.columns  # 特征名称


# In[30]:


from sklearn.model_selection import train_test_split  # 分离 训练集 验证集 测试集 

X_t, X_test, Y_t, Y_test = train_test_split(feature_v, label_v, test_size=0.2) # X_test 占0.2  
X_train, X_val, Y_train, Y_val = train_test_split(X_t, Y_t, test_size=0.25) # X_train 占 0.6 X_val 占0.2


# In[31]:


len(X_train), len(X_val), len(X_test) # 训练集： 验证集(参数调整)： 测试集 =  6：2: 2


# In[34]:


X_train # np.array 数据 


# In[35]:


from sklearn.externals import joblib 

# joblib.dump(clf, name) 保存模型
# joblib.load(clf) 读取模型

from sklearn.externals.six import StringIO
# 
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import accuracy_score, recall_score, f1_score 


# In[36]:



models=[]

models.append(("KNN", KNeighborsClassifier(n_neighbors=3))) # KNN 近邻数为3
models.append(("GaussianNB", GaussianNB())) # 高斯贝叶斯
models.append(("BernoulliNB", BernoulliNB())) # 伯努利贝叶斯
models.append(("DecisionTreeGini", DecisionTreeClassifier())) # 默认基尼不纯度的决策树
models.append(("DecisionTreeEntropy", DecisionTreeClassifier(criterion='entropy'))) # 熵增益的决策树
models.append(("SVM Classifier", SVC(C=1000,gamma='scale'))) # SVC 
models.append(("RadomForest", RandomForestClassifier())) # 并联投票的随机森林
models.append(("OriginalRandomForest", RandomForestClassifier(n_estimators=11, max_features=None))) # 11棵树的森林 每棵树选择特征全部
models.append(("Adaboost", AdaBoostClassifier(n_estimators=100))) # 100个弱分类器串联分权的集成方法
models.append(("LogisticRegression", LogisticRegression(C=1000, tol=1e-10, solver="sag", max_iter=10000))) # 逻辑回归-随机梯度下降-最大迭代1000
models.append(("GBDT", GradientBoostingClassifier(max_depth=6, n_estimators=100))) # 集成方法 GBDT 梯度提升树


# In[37]:


import re,random,time
import pydotplus

result_lst = pd.DataFrame()

for clf_name, clf in models:
    clf.fit(X_train, Y_train)  # 生成模型
    xy_lst = [(X_train,Y_train),(X_val,Y_val)] 

    # 绘制决策树
    if re.search(str.lower('tree'), str.lower(clf_name)):
        dot_data = StringIO()
        export_graphviz(clf, out_file=dot_data,feature_names=feature_names,
                                       class_names=["Y","N"],
                                       filled=True,rounded=True,
                                       special_characters=True)
        graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
        file_name = clf_name + time.strftime('%H%M') + ".pdf"
        graph.write_pdf(file_name)

    for i in  range(len(xy_lst)):
        X_part = xy_lst[i][0]
        Y_part = xy_lst[i][1]
        Y_pred = clf.predict(X_part) # 预测模型
        if i %2 == 0 :
            add =  str("train") + "-"
        else:
            add =  str("val") + "-"
        result_lst.at[clf_name, add+"ACC"] = accuracy_score(Y_part,Y_pred) # 正确率
        result_lst.at[clf_name, add+"REC"] = recall_score(Y_part,Y_pred) # 召回率
        result_lst.at[clf_name, add+'F1'] = f1_score(Y_part,Y_pred)  # F分数

        result_lst = result_lst.sort_index(axis=1) # 对columns排序 将相同指标放在一起

result_lst


# In[41]:


plt.figure(figsize=(8,6))
sns.set_context('notebook')
sns.heatmap(result_lst,annot=True)

