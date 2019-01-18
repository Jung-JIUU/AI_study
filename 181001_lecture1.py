#!/usr/bin/env python
# coding: utf-8

# In[1]:


a=73
b=12

print("a+b =", a+b) # 더하기
print("a-b =", a-b) # 빼기
print("a*b =", a*b) # 곱하기
print("a/b =", a/b) # 나누기 
print("a//b =", a//b) # 몫
print("a%b =", a%b) # 나머지


# In[4]:


import sys
print("Python버전:{}".format(sys.version))

import pandas as pd
print("pandas버전:{}".format(pd.__version__))

import matplotlib as plt
print("matplot버전:{}".format(plt.__version__))

import numpy as np
print("numpy버전:{}".format(np.__version__))

import matplotlib as plt
print("matplot버전:{}".format(plt.__version__))

import scipy as sp
print("scipy버전:{}".format(sp.__version__))

import IPython
import sklearn
print("IPython:{}".format(IPython.__version__))
print("sklearn:{}".format(sklearn.__version__))


# In[1]:


import math  
x = 10.2
print(math.ceil(x)) # 올림
print(math.floor(x)) # 내림
print(math.factorial(4)) # 4! == 4 * 3 * 2 * 1
print(sum([1,2,3])) #자연수에 강하다 
print(math.fsum([.1,.1,.1,.1,.1,.1,.1,.1,.1])) #


# In[2]:


import numpy as np
x= np.array([[1,2,3],[4,5,6]],np.int32)  #ndarray!!!!!
type(x) #numpy.ndarray
x.dtype #('int32')
x.shape #(2,3)
x[0,2]


# In[17]:


import numpy as np
import matplotlib.pyplot as plt
x = np.arange(0,5,0.5) #0부터5까지 0.5 간격으로
y = np.sin(x)
plt.plot(x,y)
plt.show()

y = np.cos(x)
plt.plot(x,y)
plt.show


# In[10]:


import pandas as pd
df = pd.DataFrame({
    "name":["이은정","심준","정지우"],
    "age":[60,28,27]
})

print(df)

df["age_plus_one"] = df["age"]+1
df["age_time_two"] = df["age"]*2
df["age_squared"] = df["age"]*df["age"]
df["over_30"] = (df["age"]> 30)


# In[3]:


import pandas as pd

s = pd.Series([1,2,3])
print(s)

s2 = s+2
print(s2.index) #데이터의 합

s3 = s+ pd.Series([4,5,6])
print(s3)

df = pd.DataFrame(s3)
df


# In[9]:


import pandas as pd  #조인
df1 = pd.DataFrame({ '고객번호':[1001,1002,1003,1004,1005,1006,1007], 
                    '이름':['둘리','도우너','또치','길동','희동','마이콜','영희'] }, columns = ['고객번호','이름'])
print(df1)
print("\n ____________")

df2 = pd.DataFrame({'고객번호':[1001,1001,1005,1006,1008,1001], 
                    '금액':[10000,20000,15000,5000,100000,30000]},columns = ['고객번호','금액'])
print(df2)
print("\n ____________")
df3 = pd.merge(df1,df2)
print(df3)
print("\n ____________")
df3 = pd.merge(df1, df2, how = 'outer')
print(df3)
print("\n ____________")
df3 = pd.merge(df1, df2, how = 'left')
print(df3)
print("\n ____________")
df3 = pd.merge(df1, df2, how = 'right')
print(df3)


# In[18]:


import pandas as pd

df1 = pd.DataFrame({'이름':['영희','철수','철수'],
                   '성적':[1,2,3]})
print(df1)

df2 = pd.DataFrame({'성명':['영희','영희','철수'], #조인할 컬럼이 없을때 지정해서 잡아준다
                   '성적2':[4,5,6]})
print(df2)

df3 = pd.merge(df1,df2, left_on='이름', right_on='성명')
print(df3)


# In[7]:


import pandas as pd
import numpy as np

df1 = pd.DataFrame({
    '도시':['서울','서울','서울','부산','부산'],
    '연도':[2000,2005,2010,2000,2005],
    '인구':[9853972,9762546,9631482,3655437,3512547]})

df2 = pd.DataFrame(
np.ndarray(12).reshape((6,2)),
index = [['부산','부산','서울','서울','서울','서울'],[2000,2005,2000,2005,2010,2015]], columns = ['데이터1','데이터2'])
df3 = pd.merge(df1, df2,left_on=['도시','연도'], right_index=True) #도시하나만 잡으면 안잡힌다


# In[8]:


import pandas as pd

df1 = pd.DataFrame(
[[1.,2.],[3.,4.],[5.,6.]],
index = ['a','b','c'],
columns = ['서울','부산'])
print(df1)

df2 = pd.DataFrame(
[[7.,8.],[9.,10.],[11.,12.],[13.,14.]], #인덱스끼리 조인을 걸어준다
index = ['b','c','d','e'],
columns = ['대구','광주'])
print(df2)

df3 = pd.merge(df1,df2,how = 'outer',
              left_index =True, right_index = True)
print(df3)


# In[24]:


p = "이름: %s 나이:%d" %("김은지",25)
print(p)

p = "X = %0.3f, Y= %10.3f" %(3.141592,3.141592)
print(p)


# In[26]:


s = ','.join(['가나','다라','마바'])
print(s)

items = s.split(',')
print(items)

deaparture, _, arrival = "Seattle-Seoul".partition('-')
print(deaparture)


# In[77]:


s = "Name:{0}, Age:{1}".format("김민석",25) #위치를 기준으로 한 포맷팅
print(s)

s = "Name:{name}, Age:{age}".format(name="김민석",age=25) #필드명을 기준으로 한 포맷팅
print(s)

area = (10, 20)
s= "width : {x[0]}, height : {x[1]}".format(x = area) #인덱스 혹은 키를 사용하여 포맷팅
print(s)


# In[ ]:





# In[59]:


import pandas as pd  #조인
df2 = pd.DataFrame({ '고객번호':[1001,1001,1003,1004,1005,1006,1007,1008,1009,1001], 
                    '이름':['둘리','도우너','또치','길동','희동','마이콜','영희','반희','잔나','춘추'],
                   '금액':[10000,10000,15000,5000,100000,30000,20000,10000,30000,40000]}, columns = ['고객번호','이름','금액'])

df3 = pd.DataFrame({ '고객번호':[1001,1003,1004,1005,1006,1007], 
                    '이름':['둘리','또치','길동','희동','마이콜','영희'],
                   '금액':[10000,15000,5000,100000,30000,20000],
                   '센':[0,0,0,0,0,0]}, columns = ['고객번호','이름','금액','센'])


# In[67]:


df2


# In[70]:


pd.merge(df2,df3,how='left')


# In[55]:


df3


# In[24]:


x = df3.update(df4)
x


# In[18]:


df3 = pd.merge(df1, df2, how = 'outer')
df3
df4 = df3.sort_values(by=['이름'])
df4

