import pandas as pd
import numpy as np
from scipy import stats
from numpy.linalg import inv
import matplotlib.pylab as plt
import os

os.chdir('e:/MIT4/6.439/pset1')
# 1.2
df_gamma = pd.read_csv('data/gamma-ray.csv')
df_gamma.head()
lam = np.sum(df_gamma['count'])/np.sum(df_gamma['seconds'])
lam



df_gamma['count'] = pd.to_numeric(df_gamma['count'], downcast='signed')
df_gamma['count'].dtype


df_gamma['x!'] = df_gamma['count'].apply(np.math.factorial)


# 1.4
df_golub = pd.read_csv('data/golub_data/golub.csv', index_col=0)
df_cl = pd.read_csv('data/golub_data/golub_cl.csv' ,index_col=0)
df_names = pd.read_csv('data/golub_data/golub_gnames.csv', index_col=0)

df_golub.columns = list(range(1,39))
# 3051 genes of 18 patients
df_golub
df_cl
df_cl
(1-df_cl)['x'].sum()
# How many genes are associated with the different tumor types (meaning that their expression level differs between the two tumor types) using (i) the uncorrected p-values, (ii)the Holm-Bonferroni correction, and (iii) the Benjamini-Hochberg correction?

# split the data into ALL and AML
col_ALL = (df_cl.loc[df_golub.columns] == 0).transpose().values.tolist()[0]
col_AML = (df_cl.loc[df_golub.columns] == 1).transpose().values.tolist()[0]
df_ALL = df_golub[df_golub.columns[col_ALL]]
df_AML = df_golub[df_golub.columns[col_AML]]
df_ALL
# hypothesis testing
from scipy.stats import ttest_ind


# test each gene
t_stat, p_value = [],[]
for i in range(df_golub.shape[0]):
    # t, p =  ttest_ind( df_ALL.iloc[i], df_AML.iloc[i])
    t, p =  ttest_ind( df_ALL.iloc[i], df_AML.iloc[i], equal_var=False )
    t_stat.append(t)
    p_value.append(p)

df_welch_ttest = pd.DataFrame(index = range(1, df_golub.shape[0]+1))
df_welch_ttest['t_stat'] = t_stat
df_welch_ttest['p_value'] = p_value

df_welch_ttest
df_welch_ttest['significant_uncorrected'] = df_welch_ttest['p_value']<0.05
pvals = df_welch_ttest['p_value']

# bonferroni correction
def holm_bonferroni(pvals, alpha=0.05):
    m, pvals = len(pvals), np.asarray(pvals)
    ind = np.argsort(pvals)
    test = [p > alpha/(m+1-k) for k, p in enumerate(pvals[ind])]
    significant = np.zeros(np.shape(pvals), dtype='bool')
    significant[ind[0:m-np.sum(test)]] = True
    return significant

# Benjamini-Hochberg procedure
def BH(pvals, q=0.05):
    m = len(pvals)
    significant = np.zeros(m, dtype='bool')
    sort_ind = np.argsort(pvals).astype(int)+1 # sort the p-values

    for i in range(1,m+1): #i = the individual p-value’s rank
         if pvals[sort_ind[i]] < (i)*q/m:
            significant[sort_ind[i]-1] = True # record the significant index
    return significant

significant_pvals = holm_bonferroni(pvals, alpha=0.05)
df_welch_ttest['significant_pvals'] = significant_pvals
significant_pvals_BH = BH(pvals, q=0.05)
df_welch_ttest['significant_pvals_BH'] = significant_pvals_BH
df_welch_ttest


df_welch_ttest.sum()


# 1.6
syn_x = pd.read_csv('data/syn_X.csv',header=None)
syn_y = pd.read_csv('data/syn_Y.csv',header=None)

syn_x.columns = ['x1','x2']
syn_x['x0'] = 1

syn_x = syn_x[['x0','x1','x2']]
X = syn_x.values
Y = syn_y.values
X.shape
#beta = (X'X)^-1X'Y
a = np.matmul(inv(np.matmul(np.transpose(X),X)),np.transpose(X))

beta = np.matmul(a,Y)

beta

def gradientDescent(X, Y, beta_0, alpha, t):
    m, n = X.shape # m is number of cases, n is the number of variables
    cost = pd.DataFrame(np.zeros([t,2]))
    cost.columns = ['step','cost']
    beta = beta_0

    for i in range(t):
        # vectorized gradient: X'*(Y-X*beta)
        res = Y- np.matmul(X, beta)
        beta = beta + 2 * alpha * (1/m) * np.matmul(np.transpose(X), res)
        # calculate the cost base on current beta
        cost['step'][i] = i
        cost['cost'][i] = calCost(X, Y, beta)

    cost.plot(kind = 'scatter', x = 'step',y = 'cost')
    return beta, cost

def calCost(X, Y, beta):
    m, n = X.shape
    # vectorized cost: (X*beta - Y)'(X*beta - Y)
    residual = Y- np.matmul(X, beta)
    return (1/(2*m))*np.matmul(np.transpose(residual), residual)

beta_0 = np.matrix('0.5; 0.5; 0.5')
alpha = 0.1
t = 50
beta, cost = gradientDescent(X, Y, beta_0,alpha, t )


beta_0 = np.matrix('0.5; 0.5; 0.5')
alpha = 0.01
t = 50
beta, cost = gradientDescent(X, Y, beta_0,alpha, t )

beta_0 = np.matrix('0.5; 0.5; 0.5')
alpha = 0.05
t = 50
beta, cost = gradientDescent(X, Y, beta_0,alpha, t )

beta_0 = np.matrix('0.5; 0.5; 0.5')
alpha = 0.8
t = 50
beta, cost = gradientDescent(X, Y, beta_0,alpha, t )

beta_0 = np.random.rand(3,1)
alpha = 0.05
t = 100
beta, cost = gradientDescent(X, Y, beta_0,alpha, t )

df_mort_0 =  pd.read_csv('data/mortality.csv')
df_mort = df_mort_0


# check the scatterplot


df_mort.head()
# check the correlation matrix
import seaborn as sns

corr = df_mort.iloc[:,2:].corr()
corrplot = sns.heatmap(corr,xticklabels=corr.columns,yticklabels=corr.columns, linewidths=.01,cmap="YlGnBu")
fig = corrplot.get_figure()
fig.savefig('corrplot.png')



df_mort.columns
df_mort = df_mort.drop(['HC'], axis = 1) # drop the intercorrelated

sns.set(style="ticks")
pairplot = sns.pairplot(df_mort.iloc[:,1:], diag_kind="kde",markers="+",plot_kws=dict(s=50, edgecolor="b", linewidth=1),diag_kws=dict(shade=True))

pairplot.savefig('pairplot1.png')
# log-transformation
df_mort[['SO2','NOx','Pop']] = np.log(df_mort[['SO2','NOx','Pop']])
df_morthead()
# normalize the Data
data = df_mort.iloc[:,1:]
data = (data - data.mean())/(data.max() - data.min())
df_mort.iloc[:,1:] = data


df_mort.plot(kind = 'bar', x = 'City', y = 'Mortality',fontsize = 5)
plt.savefig('city.pdf')
df_mort[df_mort['Mortality'] == df_mort['Mortality'].max()]
df_mort[df_mort['Mortality'] == df_mort['Mortality'].min()]


# GD on raw data
Y_r = pd.DataFrame(df_mort_0['Mortality']).values
X_r = df_mort_0.iloc[:,2:].values
Y_r.shape
m, n = X_r.shape

beta_0 = np.random.rand(n,1)
beta_0


alpha = 0.00001
t = 1000
beta, cost = gradientDescent(X_r, Y_r, beta_0, alpha, t )
cost



Y= pd.DataFrame(df_mort['Mortality']).values
X = df_mort.iloc[:,2:].values
m, n = X.shape
beta_0 = np.random.rand(n,1)
alpha = 0.05
t = 2000
beta, cost = gradientDescent(X, Y, beta_0, alpha, t )
beta

#plot the residual
import scipy.stats as stats
import pylab



residuals = np.transpose(Y - np.matmul(X, beta))
res_list = sorted(residuals[0].tolist())
def q_q_plot(data):
    norm=np.random.normal(0,2,len(data))
    norm.sort()

    plt.plot(norm,data,"o")
    z = np.polyfit(norm,data, 1)
    p = np.poly1d(z)
    plt.plot(norm,p(norm),"k--", linewidth=2)
    plt.title("Normal Q-Q plot", size=20)
    plt.xlabel("Theoretical quantiles", size=18)
    plt.ylabel("Expreimental quantiles", size=18)
    plt.tick_params(labelsize=16)
    plt.savefig('qqplot.png')
    plt.show()

q_q_plot(res_list)


#
