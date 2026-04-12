import math 


def Fejer(x,j):
    tab = [];
    if x==0:
        return [0]*(j+1)
    for i in range(0,j+1):
        tab.append(math.pow(math.pi, -1) * math.pow(math.sin(0.5 * (i + 1) * x) / (math.sin(x / 2.0)), 2) / (2.0 * (i + 1)))
    return tab;

def Dirichlet(x,j):
    tab = [];
    if(j>=0):
        tab.append(0.5);
    if(j>0):
        for i in range(1,j+1):
            tab.append(math.sin((i + 0.5) * x) / (2*math.sin(x / 2.0)))
    return tab;

def Laguerre(x,j):
    tab = [];
    if(j>=0):
        tab.append(math.pow(math.e, -x / 2));
    if (j>=1):
        tab.append(tab[0] * (1 - x));
    if (j > 1):
        for i in range(2,j+1):
            tab.append(((2 * (i - 1) + 1 - x) * tab[i - 1] - (i - 1) * tab[i - 2]) / i);
    return tab;

def Hermite(x,j):
    tab = [];
    if(j>=0):
        tab.append(math.pow(math.pi, -0.25) * math.exp(-x * x / 2.0))
    if(j>=1):
        tab.append(-math.sqrt(2.0) * x * tab[0])
    if (j > 1):
        for i in range(2,j+1):
            tab.append(-math.sqrt(2.0 / (i)) * x * tab[i - 1] - math.sqrt((i - 1.0) / i) * tab[i - 2]);
    return tab;

def Legendre(x,j):
    tab = [];
    if(j>=0):
        tab.append(math.pow(0.5, 0.5))
    if(j>=1):
        tab.append(math.pow(1.5, 0.5) * x);
    if(j>1):
        for i in range(2,j+1):
            tab.append((math.sqrt((2 * (i - 1) + 1) * (2 * (i - 1) + 3.0)) * x * tab[i - 1] - math.sqrt((2 * (i - 1) + 3) / (2.0 * (i - 1) - 1.0)) * (i - 1) * tab[i - 2]) / i);
    return tab;

choices = {"Fejer": Fejer, "Dirichlet": Dirichlet, "Laguerre": Laguerre, "Hermite": Hermite, "Legendre": Legendre}
