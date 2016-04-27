f = open('census-income.data','r')
g = open('x','a')
for a in f:
    if '50000+' in a:
        g.write(a)

