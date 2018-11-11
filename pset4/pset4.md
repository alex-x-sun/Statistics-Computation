# Problem Set 4

```Python
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import norm
from numpy.linalg import inv
import matplotlib.pyplot as plt
import os
from pathlib import Path
import re
import statistics as stat
import seaborn as sns
import random
import networkx as nx

os.chdir('e:/MIT4/statistics-Computation/pset4')
```

## 4.2  Investigating a time-varying criminal network
In this problem, you will study a time-varying criminal network that is repeatedly disturbed by police forces. The data for this problem can be found in CAVIAR.zip.

### (a)
For each of the 11 phases, compute and list the:
• degree,
• betweenness centrality
• eigenvector centrality of the actors under investigation.

#### degree
```Python
# the list of actors under investigation
under_investigation = [1,3,83,86,85,6,11,88,106,89,84,5,8,76,77,87,82,96,12,17,80,33,16]
# build the network
n = 110
phases = 11
graphs = {}

for p in range(phases):
    file_name = "data/phase" + str(p+1) + ".csv"
    data = pd.read_csv( file_name, header = 0, index_col= 0)
    actors = data.index.values.tolist()
    M = np.zeros((n+1, n+1))
    for i in actors:
        for j in actors:
            M[i,j] = data[str(j)][i]
    G = nx.DiGraph(M)
    G.remove_node(0)
    graphs[p+1] = G

# degree
in_degree_df = pd.DataFrame(0, index = under_investigation, columns = range(1, phases+1))

for p in range(1, phases+1):
    in_degrees = graphs[p].in_degree(nbunch = under_investigation)
    for i in under_investigation:
        in_degree_df.loc[i, p] = in_degrees[i]
# in_degree_df

out_degree_df = pd.DataFrame(0, index = under_investigation, columns = range(1, phases+1))

for p in range(1, phases+1):
    out_degrees = graphs[p].out_degree(nbunch = under_investigation)
    for i in under_investigation:
        out_degree_df.loc[i, p] = out_degrees[i]
# out_degree_df

```
in-degrees:<br/>
![id](/pset4/figure/id.PNG)

out-degrees:<br/>
![od](/pset4/figure/od.PNG)

#### Betweeness Centrality
```Python
between_df = pd.DataFrame(0, index = under_investigation, columns = range(1, phases+1))
for p in range(1, phases+1):
    b_cen = nx.betweenness_centrality(graphs[p])
    for i in under_investigation:
        between_df.loc[i, p] = b_cen[i]
# between_df
```
betweeness centrality:\n
![bc](/pset4/figure/bc.PNG)

#### Eigenvector Centrality
```Python
eigen_df = pd.DataFrame(0, index = under_investigation, columns = range(1, phases+1))
for p in range(1, phases+1):
    e_cen = nx.eigenvector_centrality(graphs[p], max_iter = 1000)
    for i in under_investigation:
        eigen_df.loc[i, p] = e_cen[i]
# eigen_df
eigen_l_df = pd.DataFrame(0, index = under_investigation, columns = range(1, phases+1))
reverse_g = {}
for p in range(1, phases+1):
    reverse_g[p] = graphs[p].reverse()
    e_lcen = nx.eigenvector_centrality(reverse_g[p], max_iter = 1000)
    for i in under_investigation:
        eigen_l_df.loc[i, p] = e_lcen[i]
eigen_l_df
```
Table of right eigenvector centrality:<br/>
![ec](/pset4/figure/ec.PNG)
Table of left eigenvector centrality:<br/>
![ecl](/pset4/figure/ecl.PNG)

### (b)
Describe which actors are central and which actors are only peripheral. Explain and validate
your reasoning. Feel free to compute other graph parameters, in addition to the centrality
measures in (a), to aid you in validating your answer. Who seem to be the three principal
traffickers?

```Python
# We sort the degree and centrality measures of (a):
in_degree_df.mean(1).sort_values(ascending = False)[:5]
out_degree_df.mean(1).sort_values(ascending = False)[:5]
between_df.mean(1).sort_values(ascending = False)[:5]
between_df.mean(1).sort_values(ascending = False)[-10:]
eigen_df.mean(1).sort_values(ascending = False)[:10]
eigen_l_df.mean(1).sort_values(ascending = False)[:10]

```
The betweeness centrality can grasp the "connection" of an actor in a criminal network. The top 5 central players are: n1, n12, n3, n87, n76. We noticed that They also have high degrees and eigenvector centrality values. The peripheral actors are those with low betweeness centrality values. The min value of betweeness centrality is 0, and the corresponding actors are: n16, n106, n33, n77, n17, n80, n5.

The eigenvector centrality can measure an actor's connection with other important actors. As the edge in our directed graph means a phone call, we think the left and right eigenvalue centrality could mean different types of "importance". But in our case, the top players usually have high score in both. They are n1, n3, n85, and n76.

However, all these measures cannot find the real central actor without further information about the behavioral and organizational pattern of this criminal network. It's possible that the bosses will call others to give orders but never receive calls, or only receive calls but issue orders in another way, or simply do not use telephone at all and do business through their agents. We can only know n1, n3, n12, n87, n76, n85 are important traffickers.

### (c)
Are there other actors that play an important role but are not on the list of investigation? List them, and explain why they are important.

```Python
## repeat (a) and include all
# degree
in_degree_alldf = pd.DataFrame(0, index = range(1, n+1), columns = range(1, phases+1))
out_degree_alldf = pd.DataFrame(0, index = range(1, n+1), columns = range(1, phases+1))
between_alldf = pd.DataFrame(0, index = range(1, n+1), columns = range(1, phases+1))
eigen_alldf = pd.DataFrame(0, index = range(1, n+1), columns = range(1, phases+1))
eigen_l_alldf = pd.DataFrame(0, index = range(1, n+1), columns = range(1, phases+1))


for p in range(1, phases+1):
    in_degrees = graphs[p].in_degree()
    out_degrees = graphs[p].out_degree()
    b_cen = nx.betweenness_centrality(graphs[p])
    e_cen = nx.eigenvector_centrality(graphs[p], max_iter = 1000)
    e_lcen = nx.eigenvector_centrality(reverse_g[p], max_iter = 1000)

    for i in range(1, n+1):
        in_degree_alldf.loc[i, p] = in_degrees[i]
        out_degree_alldf.loc[i, p] = out_degrees[i]
        between_alldf.loc[i, p] = b_cen[i]
        eigen_alldf.loc[i, p] = e_cen[i]
        eigen_l_alldf.loc[i, p] = e_cen[i]

in_degree_alldf.mean(1).sort_values(ascending = False)[:5]
out_degree_alldf.mean(1).sort_values(ascending = False)[:5]
between_alldf.mean(1).sort_values(ascending = False)[:10]
between_df.mean(1).sort_values(ascending = False)[:10]
eigen_alldf.mean(1).sort_values(ascending = False)[:10]
eigen_l_alldf.mean(1).sort_values(ascending = False)[:10]
eigen_l_df.mean(1).sort_values(ascending = False)[:10]

np.setdiff1d(between_df.mean(1).sort_values(ascending = False)[:10].index.values.tolist(), between_alldf.mean(1).sort_values(ascending = False)[:10].index.values.tolist())
np.setdiff1d(eigen_df.mean(1).sort_values(ascending = False)[:10].index.values.tolist(), eigen_alldf.mean(1).sort_values(ascending = False)[:10].index.values.tolist())
np.setdiff1d(eigen_l_df.mean(1).sort_values(ascending = False)[:10].index.values.tolist(), eigen_l_alldf.mean(1).sort_values(ascending = False)[:10].index.values.tolist())

```
The n8, n83, n89 are in betweeness centrality top 10 list of all actors but not investigated.
The n88 is in right eigenvector centrality top 10 list of all actors but not investigated.
The n6, n11 are in left eigenvector centrality top 10 list of all actors but not investigated.
### (d)
Describe the coarse pattern(s) you observe as the network evolves through the phases. Does
the network evolution reect the background story? Explain.

```Python
# track the changes of the network
change_df = pd.DataFrame(0, index = range(1, phases+1), columns = ['phase', 'size','density'])
change_df['phase'] = range(1, phases+1)
for p in range(1, phases+1):

    change_df.loc[p, 'size'] =  graphs[p].size(weight = 'weight')
    change_df.loc[p, 'density'] = nx.density(graphs[p])

# change_df
import seaborn as sns; sns.set()
ax1 = sns.lineplot(x = "phase", y = "size", data = change_df)
ax2 = sns.lineplot(x = "phase", y = "density", data = change_df)
fig1 = ax1.get_figure()
fig1.savefig('figure/sizechange.png')
fig2 = ax2.get_figure()
fig2.savefig('figure/densechange.png')
```
Plot of network size change:<br/>
![sc](/pset4/figure/sizechange.png)<br/>
Plot of network density change:<br/>
![dc](/pset4/figure/densechange.png)<br/>

The network size and density decreased a lot between phase 4 and phase 5. It is caused by the seizure in phase 4. Phase 5 had no seizure, which caused the increase in network size and density in phase 6. The fluctuations after phase 6 were caused by seizures.


### (e)
Describe and interpret the evolution of the role of the central actors found in (b). At which phases are they active? When do they withdraw? Find indices in the network evolution that reflect the description given to them.

```Python
central_actors = [1,3,12,87,76,85]

central_actors_bc = between_df.loc[central_actors]
central_actors_ec = eigen_df.loc[central_actors]
central_actors_elc = eigen_l_df.loc[central_actors]


# fig, ax = plt.subplots()
ax3 = central_actors_bc.T.plot()
ax3.set(xlabel = "Phases", ylabel = "Betweenness Centrality")
ax3.set_xticks(range(11))
ax3.set_xticklabels(central_actors_bc.columns)
plt.savefig('figure/central_actors_bc.png', dpi = 300)
plt.show()

ax4 = central_actors_ec.T.plot()
ax4.set(xlabel = "Phases", ylabel = "Right Eigenvector Centrality")
ax4.set_xticks(range(11))
ax4.set_xticklabels(central_actors_ec.columns)
plt.savefig('figure/central_actors_ec.png', dpi = 300)
plt.show()

ax5 = central_actors_elc.T.plot()
ax5.set(xlabel = "Phases", ylabel = "Left Eigenvector Centrality")
ax5.set_xticks(range(11))
ax5.set_xticklabels(central_actors_elc.columns)
plt.savefig('figure/central_actors_elc.png', dpi = 300)
plt.show()

```
Plot of betweeness centrality change of selected central actors:<br/>
![cbc](/pset4/figure/central_actors_bc.png)<br/>
Plot of right eigenvector centrality change of selected central actors:<br/>
![cec](/pset4/figure/central_actors_ec.png)<br/>
Plot of left eigenvector centrality change of selected central actors:<br/>
![celc](/pset4/figure/central_actors_elc.png)<br/>

From the plot, we can see that n1 was in central position throughout the 11 phases, although we observed a drop in betweeness centrality in phase 9. n3 and n12's centrality dropped in phase 7 but increased after phase 7. n76 was steadly becoming more central, especially since phase 6. n87 started to become central since phase 8. n85 had a drop in phase 3 but generally highly connected with important actors.

### (f)
Examine the frequency and the directions of the communications of (n1) as the network evolves. Any contrast or pattern(s) you observe? Describe, explain and interpret.

```Python
n1_df = pd.DataFrame(0, index = range(1, phases+1), columns = ['in degree', 'out degree','degree'])
n1_df.loc[1,'out degree']

for p in range(1, phases+1):
    n1_df.loc[p,'degree'] = graphs[p].degree(1)
    n1_df.loc[p, 'in degree'] = in_degree_df.loc[1,p]
    n1_df.loc[p,'out degree'] = out_degree_df.loc[1,p]
# n1_df

ax6 = n1_df.plot()
ax6.set(xlabel = "Phases", ylabel = "Degrees of n1")
ax6.set_xticks(range(1,12))
ax6.set_xticklabels(n1_df.index)
plt.savefig('n1d.png', dpi = 300)
plt.show()

```
We plot the in and out degree changes of n1:<br/>
![n1d](/pset4/figure/n1d.png)<br/>
Generally, phone calls from n1 (out degree) were more than calls to n1 (in degree).In phase 6, n1 received more calls than making calls to others. His communication frequency dropped a lot in phase 9 corresponding to 2 seizures.

### (g)
Would you consider that the particular strategy adopted by the police had an impact on the criminal network throughout the different phases of the investigation? What kind of impact? Explain.

```Python
ax1 = sns.lineplot(x = "phase", y = "size", data = change_df)
ax2 = sns.lineplot(x = "phase", y = "density", data = change_df)
```
From the lineplot of network size and edge density, we found that the network is highly sensitive to seizures at first, but become less and less affected. Also, by investigating the betweeness/eigenvector centrality of the central actors, we think that the position of central actors may shift to resist the threats to the network. Therefore, the police should keep monitoring the communications of the network, make one or two seizures to disturb their organization and leadership, and pay attention to the actors with increasing centrality.



## 4.3  Co-offending Network
The data for this problem set consists of individuals who were arrested in Quebec between 2003 and 2010. Some of the individuals have always acted solo, and have been arrested alone throughout their 'career'. Others co-offended with other individuals, and have been arrested in groups. The goal of this problem set is to construct and analyze the co-oender network. The nodes in the network are the oenders, and two offenders share a (possibly weighted) edge whenever they are arrested for the same crime event.


build the whole co-offender network. Discard the isolated nodes, thus every node will have degree>=2. Given the size of the network, be careful regarding computational and memory constraints. Be sure to use sparse representations of the data whenever possible.

```Python
cooffend_df = pd.read_csv('data/Cooffending.csv').to_sparse()
cooffend_df.shape
cooffend_df.head()


# find cooffending cases
import collections

case_l = cooffend_df['SeqE'].tolist()

coofending_case_l = []
for item, count in collections.Counter(case_l).items():
    if count > 1:
        coofending_case_l.append(item)

len(coofending_case_l)
len(set(case_l))
# there were 1164836 unique cases, in which 84239 cases were co-offending cases
# remove individual cases
net_df = cooffend_df[cooffend_df['SeqE'].isin(coofending_case_l)].sort_values(by = ['SeqE'])
# net_df0 = net_df.copy()
# net_df.head(20)
# net_df = net_df0

# build the network
G = nx.Graph()

case_num = 0
offender_l = []

for i in range(len(net_df)):
    # extract one observation of offender and case
    seq = net_df['SeqE'][i]
    offender = net_df['NoUnique'][i]
    # add the offender into graph if it's not in it
    if not G.has_node(offender):
        G.add_node(offender)
    # as the data we use here is sorted by case number, we only need to go case by case
    if seq == case_num:
        # if the offender is still in current case
        for x in offender_l:
            # two possible cases: this offender co-offend other cases that have beed recorded in our graph with the same person, or this is the first recorded Cooffending

            if G.has_edge(offender, x):
                G[offender][x]['weight'] += 1
            else:
                G.add_edge(offender, x, weight = 1)
        # add this offender in the case offender list
        offender_l.append(offender)

    # if we go to another case, reset case number and offerders list
    else:
        case_num = seq
        offender_l = [offender]

nx.draw_shell(G, with_labels=True)

```

### (e)
How many nodes does the network have? How many solo offenders are there in the data set? How many (unweighted) edges does the graph contain?

```Python
num_node = nx.number_of_nodes(G)

num_solo = len(set(cooffend_df['NoUnique'])) - num_node

unweighted_size = G.size()

```
### (f)
Plot the degree distribution (or an approximation of it if needed) of the network.

```Python
coof_degree = G.degree()

def PlotDegree(G):
    degree_sequence = sorted([d for n, d in G.degree()], reverse=True)  # degree sequence
    degreeCount = collections.Counter(degree_sequence)
    deg, cnt = zip(*degreeCount.items())
    fig, ax = plt.subplots()
    plt.bar(deg, cnt, width=0.80, color='b')
    plt.title("Degree Histogram")
    plt.ylabel("Count")
    plt.xlabel("Degree")
    ax.set_xticks([d + 0.4 for d in deg])
    ax.set_xticklabels(deg)

PlotDegree(G)

```
### (g)
How many connected components does the network have?

```Python
num_compo = nx.number_connected_components(G)
```

We will now isolate the largest connected component and focus on it. This brings us down to a
more manageable size.

### (h)
How many nodes does the largest connected component have?

```Python
max_compo = max(nx.connected_components(G), key=len)
G_max_compo = max_compo
num_node_max_compo = nx.number_of_nodes(G_max_compo)
```

### (i)
Compute the degree of the nodes, and plot the degree distribution (or an approximation of it if needed) for the largest connected component. Comment on the shape of the distribution.

```Python
PlotDegree(G_max_compo)
```
### (j)
Describe the general shape of the largest connected component. Use the degree distribution from above, and compute statistics of the network to obtain an overview of its characteristics. You may want to consider the edge density, clustering, diameter, etc. Comment on the results.

```Python
nx.draw_shell(G_max_compo, with_labels=True)

den = nx.density(G_max_compo)

dia = nx.diamete(G_max_compo)

```
Thisfinal section involves some free form investigation. The following parts are optional for undergraduates.

### (k)
How many crime events are executed only by young offenders?

```Python
young_offenders = net_df[net_df['Jeunes'] == 1]['NoUnique']


```
### (l)
Investigate the relationship between young offenders and adult offenders. Study the structure of the crimes that include both, young and adult offenders. Discuss any patterns you observe.

```Python

```
### (m) 
Ask your own question, build new separate networks if needed, and get as much insight as you like. Feel free to focus on either the whole network, or the largest connected component.
```Python

```
